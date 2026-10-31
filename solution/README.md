# Rank-split LoRA

The solution keeps bitsandbytes NF4 and ordinary PyTorch dropout. For supported
PEFT `q_proj` / `v_proj` instances, it computes `H = dropout(X) @ A.T` with cuBLAS,
stores the rank-16 `H`, then fuses `H @ B.T`, scaling and addition to the base
output in one kernel. There is no full-size LoRA output or scaled-output buffer.
The defensive clone of the base output is retained.

The graph split is inspired by [LoRAFusion, section 5.1](https://arxiv.org/abs/2510.00206),
by Zhu et al. The kernels in this directory are an independent implementation.
The paper's measured speedups do not predict this implementation's results.

`lora_kernels.py` implements the forward and first-order backward. The base branch
receives `G`; the adapter computes `dB`, `dH`, `dA` and `dU`. GEMMs use cuBLAS;
a Triton kernel combines gradient dtype conversion and scaling. AMP
casts remain in the PyTorch graph, so FP32 adapter parameters receive FP32
gradients. The up output and scaling each round to the Linear compute dtype,
matching PEFT's eager operation boundaries. The incoming gradient is cast and
scaled before the backward GEMMs for the same reason.

`lora.py` wraps Soup's `_setup_transformers` after normal setup. It patches only
PEFT `Linear4bit` instances named `q_proj` or `v_proj`. The optimized path requires
NF4, a single ordinary adapter, rank 16, widths at most 1536,
at most 2048 token positions and FP16 compute (the task uses widths 1536 and 256). CPU, FP32 compute without AMP,
other shapes, variants, merged/disabled adapters and calls with extra arguments
use the saved original forward. Parameter objects and state-dict names are kept.

Dropout is called before casting for AMP. Each projection uses its own module and
the regular PyTorch generator; no seed, mask, probability or RNG state is replaced.
The NF4 base branch still computes input gradients. The
solution does not change training settings, batches, metrics, rewind logging,
optimizer, serialization, harness or task files. The shipped SwiGLU example is
still enabled; its kernels do not count as original work.

`rmsnorm.py` additionally fuses Qwen's RMSNorm forward and input backward for
frozen FP32 norm weights. Each token row reads the activation once per pass;
backward saves only the original input and inverse RMS. The cast before weight
multiplication follows the installed Qwen implementation. Trainable norm weights,
CPU, other widths and unsupported dtypes retain the original forward.

`vocab.py` streams vocabulary tiles to compute log-sum-exp, entropy, first argmax
and causal per-token CE from small partials. Training CE saves logits and row-wise
log-sum-exp; its backward writes the softmax-minus-target gradient directly in
the input dtype, avoiding a full FP32 log-softmax tensor. The narrow TRL hook
shares one vocabulary reduction across entropy, accuracy and Soup's rewind row
losses. It keeps the same attention mask, supervised mask, token counters,
distributed gathers and returned loss. Rewind reuse is tied to the exact logits
object and label identity/version, with the original routine as fallback. Other
vocabularies, objectives, adapters with virtual tokens and evaluation retain their
standard paths. No metric is disabled or approximated.

`compact.py` removes padding from frozen linear blocks and packs an entire MLP
once per block. It scatters outputs back before attention and keeps the original
attention mask and batch boundaries. Only masked positions without supervised
queries/targets may be omitted. q/v LoRA dropout still runs on full tensors with
unchanged RNG consumption. There is no packing across examples or across batches,
and no Soup training switch is changed. This uses the README's explicit permission
to remove padding inside a batch. CUDA tests cover left/right padding, valid-token
logits, loss, adapter gradients and identical generator state.
Packing/restoration and their backwards use independent Triton copy kernels;
the inverse token map makes zero-fill and scatter a single pass without atomics.
The FP16-compute MLP route also avoids promoting gate/up GEMM results to FP32
storage only to cast the activation back to FP16 for down. SwiGLU still computes
in FP32 inside the shipped example kernel; the small block output returns to the
original dtype. Gate/up input casts retain separate gradient edges.

`attention.py` keeps q/k/v and rotary embeddings compact. A custom rotary
forward/backward combines the original FP32 products and addition with no FMA.
Causal SDPA runs on each row's valid positions, preserving their order and rotary
positions. Each q/v dropout still receives the original full-sized hidden tensor,
then gathers only valid outputs. The output is restored before the residual add.
Other attention modes, cache use, dropout in attention and sliding windows retain
the original path. Tests assert the compact SDPA route is used and check RNG and
adapter gradients against the original model, in addition to separate RoPE tests.

`nf4_cache.py` reuses the exact FP16 values decoded by bitsandbytes from frozen
NF4 parameters. Each layer's cache is created lazily inside the timed process,
keyed by packed storage/version and quantization-state versions. Forward uses
the decoded matrix and input backward multiplies by the same matrix. NF4
parameters, quantization configuration and serialization stay intact; no original
unquantized checkpoint weights are substituted. This spends additional measured
VRAM to remove repeated dequantization and allocations. The cache is not written
to disk or included in adapter state dictionaries. Other compute dtypes, mutable
base weights and unsupported calls retain the original bitsandbytes path.

On the pinned Kaggle environment, from the repository root:

```bash
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py
python harness/check_rules.py
```

`test_lora.py` compares outputs and every gradient against PyTorch with nonzero B,
FP32 adapter parameters, both projection widths, irregular token counts and the
maximum 2048 tokens. It also checks real NF4/PEFT with AMP, two seeds, successive
dropout calls, identical RNG advancement, adapter save/reload, fallbacks and
idempotent setup. It asserts that the optimized path actually runs.

Local CUDA checks are development checks, not competition measurements. The local
GPU is an RTX 3050, not a T4; Windows uses a Triton development port, which is not
imported or installed by the solution. The submission uses the Kaggle environment's
ordinary `triton` package and installs nothing during training.

Local validation on 2026-10-30 used Python 3.12, torch 2.11.0+cu128, Triton
3.6.0.post26 (Windows development port), and the README-pinned Soup, transformers,
PEFT, TRL, bitsandbytes and accelerate versions:

| Check | Result |
| --- | --- |
| `python -m pytest solution -q` | 49 passed including exact token-copy gradients, RoPE and NF4-cache validation |
| `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` | 79 passed |
| Harness source analysis | 10 original kernels; no banned imports, parse errors or example lookalikes |
| `git diff --check` | Passed |
| T4 full training, quality, timing, memory and kernel share | Pending |

The initial all-Triton LoRA version (`1e50c03`) was measured on Kaggle T4:
75.40 s base vs 83.86 s solution, 14345 vs 12385 MiB, eval loss 2.04970 vs
2.04965, own kernel share 8.2%. It failed the 10% slowdown limit. Its original
LoRA matmuls consumed 3.72 s of the timed training loop. This revision returns
these GEMMs to cuBLAS, removes split-gradient buffers/reductions, widens the
fused up tiles and adds the frozen RMSNorm fusion. That revision (`b56f846`)
measured 77.30 s base vs 79.90 s solution, the same 12385 MiB solution peak,
eval loss 2.04965 vs 2.04967, and own kernel share 3.0%. It still did not provide
the required speedup or kernel share. The streaming vocabulary revision adds
loss/metrics/rewind fusion. With metadata-kwargs handling fixed (`9f24be1`), it
measured 78.60 s base vs 75.83 s solution, 14345 vs 10005 MiB, eval loss
2.04968 vs 2.04966, and 5.0% original-kernel share. All ten checks passed, but
the speed gain was under the 5% scoring threshold and the kernel share was close
to its lower bound. The subsequent compact-batch version and reduced Triton
specialization count (`fe58a4a`) measured 76.18 s base vs 64.80 s solution,
14345 vs 10445 MiB and eval loss 2.04963 vs 2.04966. It passed quality and
output comparisons but its 4.4% original-kernel share missed the 5% requirement.
Runtime NF4 caching, Triton token copies and the reduced-copy MLP route require
a new T4 measurement. That version (`097359e`) measured 75.33 s base vs 62.45 s
solution, 14345 vs 11825 MiB, eval loss 2.04968 vs 2.04964 and own kernel share
11.4%; all ten checks passed. The compact attention/rotary version and shared
copy-kernel specialization are pending measurement. The requested 45 s target
has not yet been demonstrated.

Full-command speed, peak memory, loss-curve agreement, held-out quality and the
required 5% kernel share must be measured with the unchanged harness on a Kaggle
T4. They have not been established by the local unit tests. In particular, the
LoRA-only scope may fall short of 5%; the implementation must not add unused work
or change the measurement to satisfy that threshold. Cold compilation counts,
and different GEMM/reduction orders can introduce small FP16 differences.
Higher-order autograd is unsupported. Dropout/down fusion and NF4/GEMM fusion are
not implemented.
