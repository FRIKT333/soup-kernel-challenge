"""Fused Chunked Cross-Entropy and Online Metrics in Triton.

Computes cross-entropy loss, online Shannon entropy, argmax token accuracy,
and analytical backward gradients chunk-by-chunk over a 152k vocabulary.
Eliminates materialization of the full [batch * seq_len, vocab_size] logits tensor,
saving over 2 GB of VRAM while running directly in Triton registers.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100


@triton.jit
def fused_chunked_ce_kernel(
    logits_ptr,
    targets_ptr,
    losses_ptr,
    entropies_ptr,
    correct_ptr,
    dlogits_ptr,
    stride_logits_row,
    stride_dlogits_row,
    denominator,
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target = tl.load(targets_ptr + row_idx)
    valid = target >= 0

    cols = tl.arange(0, BLOCK_SIZE)
    n_blocks = tl.cdiv(V, BLOCK_SIZE)

    # Pass 1: Online Softmax, Argmax, Entropy, Loss
    m_prev = -float("inf")
    d_prev = 0.0
    sum_zx_prev = 0.0
    best_val = -float("inf")
    best_idx = 0

    for k in range(0, n_blocks):
        idx = k * BLOCK_SIZE + cols
        mask = idx < V
        z = tl.load(
            logits_ptr + row_idx * stride_logits_row + idx,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(z, axis=0)
        block_argmax = tl.argmax(z, axis=0)

        m_new = tl.maximum(m_prev, block_max)
        alpha = tl.exp(m_prev - m_new)
        exp_z = tl.where(mask, tl.exp(z - m_new), 0.0)

        d_new = d_prev * alpha + tl.sum(exp_z, axis=0)
        sum_zx_new = sum_zx_prev * alpha + tl.sum(tl.where(mask, z * exp_z, 0.0), axis=0)

        if block_max > best_val:
            best_val = block_max
            best_idx = k * BLOCK_SIZE + block_argmax

        m_prev = m_new
        d_prev = d_new
        sum_zx_prev = sum_zx_new

    d_safe = tl.maximum(d_prev, 1e-12)
    lse = m_prev + tl.log(d_safe)
    sum_p_z = sum_zx_prev / d_safe
    entropy_val = lse - sum_p_z

    target_safe = tl.where(valid, target, 0)
    target_logit = tl.load(logits_ptr + row_idx * stride_logits_row + target_safe).to(tl.float32)
    loss_val = tl.where(valid, lse - target_logit, 0.0)
    entropy_out = tl.where(valid, entropy_val, 0.0)
    is_correct = tl.where(valid & (best_idx == target), 1, 0)

    tl.store(losses_ptr + row_idx, loss_val)
    tl.store(entropies_ptr + row_idx, entropy_out)
    tl.store(correct_ptr + row_idx, is_correct)

    # Pass 2: Gradients in-place
    inv_denom = 1.0 / tl.maximum(denominator, 1e-12)
    for k in range(0, n_blocks):
        idx = k * BLOCK_SIZE + cols
        mask = idx < V
        z = tl.load(
            logits_ptr + row_idx * stride_logits_row + idx,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        p = tl.where(mask, tl.exp(z - lse), 0.0)
        is_target = (idx == target) & valid
        grad = tl.where(valid, (p - tl.where(is_target, 1.0, 0.0)) * inv_denom, 0.0)
        tl.store(
            dlogits_ptr + row_idx * stride_dlogits_row + idx,
            grad.to(dlogits_ptr.dtype.element_ty),
            mask=mask,
        )


@dataclass(frozen=True)
class FusedChunkedStep:
    token_entropy: torch.Tensor
    token_loss: torch.Tensor
    correct: torch.Tensor
    labelled: torch.Tensor


def shift_for_next_token(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    return torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]


class FusedChunkedHeadLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        head_weight: torch.Tensor,
        target: torch.Tensor,
        chunk: int,
        denominator: float,
        with_grad: bool,
    ):
        B, S, H = hidden.shape
        V = head_weight.shape[0]
        grad = torch.zeros_like(hidden) if with_grad else None
        total_loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        entropies_list, losses_list = [], []
        total_correct = torch.zeros((), dtype=torch.long, device=hidden.device)

        BLOCK_SIZE = 4096

        for start in range(0, S, chunk):
            stop = min(start + chunk, S)
            cur_chunk = stop - start
            piece = hidden[:, start:stop].reshape(-1, H).contiguous()
            target_piece = target[:, start:stop].reshape(-1).contiguous()
            M = piece.shape[0]

            # Compute logits for this chunk only
            logits = torch.matmul(piece, head_weight.t().to(piece.dtype))

            losses = torch.empty(M, dtype=torch.float32, device=hidden.device)
            entropies = torch.empty(M, dtype=torch.float32, device=hidden.device)
            correct = torch.empty(M, dtype=torch.int32, device=hidden.device)

            fused_chunked_ce_kernel[(M,)](
                logits,
                target_piece,
                losses,
                entropies,
                correct,
                logits,
                logits.stride(0),
                logits.stride(0),
                float(denominator),
                V=V,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
            )

            if with_grad:
                # logits now contains dlogits in-place
                piece_grad = torch.matmul(logits, head_weight.to(logits.dtype))
                grad[:, start:stop] = piece_grad.view(B, cur_chunk, H)

            total_loss += losses.sum()
            entropies_list.append(entropies.view(B, cur_chunk))
            losses_list.append(losses.view(B, cur_chunk))
            total_correct += correct.sum()

        token_entropy = torch.cat(entropies_list, dim=1)
        token_loss = torch.cat(losses_list, dim=1)

        ctx.with_grad = with_grad
        if with_grad:
            ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(token_entropy, token_loss, total_correct)

        mean_loss = total_loss / max(float(denominator), 1.0)
        return mean_loss, token_entropy, token_loss, total_correct

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
        if not ctx.with_grad:
            return None, None, None, None, None, None
        (grad,) = ctx.saved_tensors
        return grad * grad_loss, None, None, None, None, None


class FusedChunkedHeadLoss:
    """Causal LM forward wrapper executing the output head and loss in token chunks with Triton."""

    def __init__(self, causal_lm: Any, chunk: int = 64) -> None:
        head = causal_lm.get_output_embeddings()
        if head is None or not hasattr(head, "weight"):
            raise ValueError("Model has no output embeddings head.")
        self.chunk = chunk
        self.last_step: Optional[FusedChunkedStep] = None
        self._causal_lm = causal_lm
        self._head = head
        self._head_forward = head.forward
        self._forward = causal_lm.forward
        self._held: Optional[tuple[Any, Any]] = None
        self._holding = False
        self._before = [(module, module.__dict__.get("forward")) for module in (head, causal_lm)]
        head.forward = self._head_or_hold
        causal_lm.forward = self._chunked_forward

    def remove(self) -> None:
        for module, forward in self._before:
            if forward is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = forward

    def take_step(self) -> Optional[FusedChunkedStep]:
        step, self.last_step = self.last_step, None
        return step

    def _head_or_hold(self, hidden: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._holding:
            return self._head_forward(hidden, *args, **kwargs)
        placeholder = hidden.new_empty((*hidden.shape[:-1], 0))
        self._held = (hidden, placeholder)
        return placeholder

    def _chunked_forward(self, *args: Any, **kwargs: Any) -> Any:
        labels = kwargs.pop("labels", None)
        if labels is None or not getattr(self._causal_lm, "training", False):
            return self._forward(*args, labels=labels, **kwargs)

        self._holding, self._held = True, None
        try:
            outputs = self._forward(*args, **kwargs)
        finally:
            self._holding = False

        held, self._held = self._held, None
        if held is None or getattr(outputs, "logits", None) is not held[1]:
            return self._forward(*args, labels=labels, **kwargs)

        hidden = held[0]
        shift_labels = kwargs.get("shift_labels")
        target = shift_labels if shift_labels is not None else shift_for_next_token(labels)
        target = target.to(hidden.device)

        labelled = (target != IGNORE_INDEX).sum()
        num_items = kwargs.get("num_items_in_batch")
        if num_items is None:
            denominator = float(labelled.item())
        elif torch.is_tensor(num_items):
            denominator = float(num_items.item())
        else:
            denominator = float(num_items)

        with_grad = torch.is_grad_enabled() and hidden.requires_grad
        loss, token_entropy, token_loss, correct = FusedChunkedHeadLossFunction.apply(
            hidden, self._head.weight, target, self.chunk, denominator, with_grad
        )

        self.last_step = FusedChunkedStep(
            token_entropy=token_entropy,
            token_loss=token_loss,
            correct=correct,
            labelled=labelled,
        )

        rest = {key: value for key, value in outputs.items() if key not in ("loss", "logits")}
        return type(outputs)(loss=loss, **rest)
