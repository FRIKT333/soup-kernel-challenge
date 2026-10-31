"""Run with python -m pytest solution/test_lora.py -q on CUDA.

Nonzero B, non-power-of-two token counts and FP32 adapter parameters are intentional.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from solution.lora import patch_model, patch_soup_lora


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _close(actual, expected):
    # Split reductions and cuBLAS sum in different orders. Scale the absolute
    # bound by RMS so cancellation near zero does not dominate weight gradients.
    rms = expected.float().square().mean().sqrt().item()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=max(3e-3, 0.003 * rms))
    error = (actual.float() - expected.float()).norm()
    assert error <= 0.003 * expected.float().norm() + 1e-5


@CUDA
@pytest.mark.parametrize("tokens,out_features", [(31, 256), (65, 1536), (257, 256),
                                                 (2048, 256), (2048, 1536)])
@pytest.mark.parametrize("base_dtype,scale,convert", [(torch.float16, 2.0, True),
                                                     (torch.float32, 0.7, False),
                                                     (torch.float32, 0.7, True)])
def test_rank_split_outputs_and_all_gradients(tokens, out_features, base_dtype, scale, convert):
    from solution.lora_kernels import rank_split_lora

    torch.manual_seed(451)
    shape = (1, tokens, 1536)
    original = [torch.randn((1, tokens, out_features), device="cuda", dtype=base_dtype),
                torch.randn(shape, device="cuda", dtype=torch.float16),
                torch.randn((16, 1536), device="cuda", dtype=torch.float32) * 0.02,
                torch.randn((out_features, 16), device="cuda", dtype=torch.float32) * 0.02]
    ref = [t.clone().requires_grad_() for t in original]
    got = [t.clone().requires_grad_() for t in original]
    base, u, a, b = ref
    # Casts reproduce AMP Linear operands while testing FP32 parameter gradients.
    h = torch.nn.functional.linear(u, a.half())
    delta = torch.nn.functional.linear(h, b.half()) * scale
    expected = base + (delta.to(base_dtype) if convert else delta)
    base, u, a, b = got
    actual = rank_split_lora(base, u, a.half(), b.half(), scale, convert)
    grad = torch.randn_like(expected) * 0.1
    expected.backward(grad)
    actual.backward(grad)
    assert actual.dtype == expected.dtype
    _close(actual, expected)
    for x, y in zip(got, ref):
        _close(x.grad, y.grad)


@CUDA
def test_strided_input_and_selective_gradients():
    from solution.lora_kernels import rank_split_lora

    torch.manual_seed(14)
    source = torch.randn((2, 1536, 33), device="cuda", dtype=torch.float16, requires_grad=True)
    u = source.transpose(1, 2)
    a = torch.randn((16, 1536), device="cuda", dtype=torch.float16) * 0.02
    b = torch.randn((256, 16), device="cuda", dtype=torch.float16) * 0.02
    base = torch.randn((2, 33, 256), device="cuda", dtype=torch.float16, requires_grad=True)
    expected = base + (u @ a.t()) @ b.t() * 2
    actual = rank_split_lora(base, u, a, b, 2)
    grad = torch.randn_like(expected)
    eg = torch.autograd.grad(expected, (source, base), grad)
    ag = torch.autograd.grad(actual, (source, base), grad)
    _close(actual, expected)
    for x, y in zip(ag, eg):
        _close(x, y)


def test_fake_model_setup_guard():
    assert patch_model(None) == 0
    assert patch_model(SimpleNamespace()) == 0


def test_setup_hook_is_idempotent_and_preserves_return(monkeypatch):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    calls = []

    def setup(self, value):
        calls.append(value)
        self.model = SimpleNamespace()
        return value

    monkeypatch.setattr(SFTTrainerWrapper, "_setup_transformers", setup)
    patch_soup_lora()
    first = SFTTrainerWrapper._setup_transformers
    patch_soup_lora()
    assert SFTTrainerWrapper._setup_transformers is first
    assert first(SimpleNamespace(), 42) == 42
    assert calls == [42]


def _peft_model(device):
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = bnb.nn.Linear4bit(1536, 1536, bias=False,
                                           compute_dtype=torch.float16, quant_type="nf4")
            self.v_proj = bnb.nn.Linear4bit(1536, 256, bias=False,
                                           compute_dtype=torch.float16, quant_type="nf4")

        def forward(self, x):
            return self.q_proj(x), self.v_proj(x)

    model = Projections().to(device)
    # AutoModel's quantized loader sets this; our synthetic model must too so
    # PEFT dispatches Linear4bit rather than its generic Linear wrapper.
    model.is_loaded_in_4bit = True
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.02)
    return model


@CUDA
@pytest.mark.parametrize("seed", [1, 829])
def test_real_nf4_peft_amp_dropout_backward_and_save(seed, tmp_path, monkeypatch):
    from peft import get_peft_model_state_dict
    from solution import lora_kernels

    calls = []
    original = lora_kernels.rank_split_lora

    def counted(*args, **kwargs):
        calls.append(args[3].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(lora_kernels, "rank_split_lora", counted)

    torch.manual_seed(21)
    ref = _peft_model("cuda").train()
    model = copy.deepcopy(ref)
    before = {name: id(p) for name, p in model.named_parameters()}
    assert patch_model(model) == 2
    assert patch_model(model) == 0
    assert before == {name: id(p) for name, p in model.named_parameters()}
    original_x = torch.randn((2, 33, 1536), device="cuda", dtype=torch.float32)
    # Check a new dropout mask on the second call, in addition to repeatable seeds.
    results = []
    states = []
    for current in (ref, model):
        x = original_x.clone().requires_grad_()
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.float16):
            first = current(x)
            second = current(x)
            loss = sum(t.float().square().mean() for t in first + second)
        loss.backward()
        results.append((first, second, x.grad,
                        {n: p.grad.clone() for n, p in current.named_parameters() if p.requires_grad}))
        states.append(torch.cuda.get_rng_state())
    assert torch.equal(*states)
    assert calls == [1536, 256, 1536, 256]
    for group in (0, 1):
        for actual, expected in zip(results[1][group], results[0][group]):
            _close(actual, expected)
    assert not torch.equal(results[1][0][0], results[1][1][0])
    _close(results[1][2], results[0][2])
    for name, grad in results[1][3].items():
        _close(grad, results[0][3][name])
    # State names, shapes and ordinary PEFT reload must survive the instance hook.
    model.save_pretrained(tmp_path)
    reloaded = copy.deepcopy(ref)
    reloaded.load_adapter(tmp_path, adapter_name="roundtrip")
    saved = get_peft_model_state_dict(model)
    loaded = get_peft_model_state_dict(reloaded, adapter_name="roundtrip")
    assert saved.keys() == loaded.keys()
    for name in saved:
        torch.testing.assert_close(saved[name], loaded[name], rtol=0, atol=0)


@CUDA
def test_peft_fallback_modes():
    model = _peft_model("cuda").eval()
    ref = copy.deepcopy(model)
    assert patch_model(model) == 2
    x = torch.randn((1, 7, 1536), device="cuda")
    # FP32 without AMP uses stock PEFT, as does disabled-adapter inference.
    for current in (model, ref):
        with torch.no_grad():
            current(x)
    for a, b in zip(model(x), ref(x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with model.disable_adapter(), ref.disable_adapter(), torch.no_grad():
        for a, b in zip(model(x), ref(x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
