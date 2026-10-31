import pytest
import torch
import triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('vocab', [73, 4099, 65537, 151936])
@pytest.mark.parametrize('ignored', [False, True])
def test_online_metrics_gradient_ignored_rows_and_argmax_ties(vocab, ignored):
    from solution.online_head import online_vocabulary_gradient_kernel
    torch.manual_seed(177)
    rows = 17
    logits = torch.randn(rows, vocab, device='cuda', dtype=torch.float16) * 0.5
    logits[0, 1] = 10.0; logits[0, 4] = 10.0
    labels = torch.randint(vocab, (rows,), device='cuda')
    labels[::3] = -100
    if ignored: labels.fill_(-100)
    ce = torch.empty(rows, device='cuda'); ent = torch.empty_like(ce)
    pred = torch.empty(rows, device='cuda', dtype=torch.int64)
    grad = torch.empty_like(logits)
    online_vocabulary_gradient_kernel[(rows,)](logits, labels, ce, ent, pred, grad, rows, vocab, 32, 2048,
        num_warps=4, enable_fp_fusion=False)
    lp = logits.float().log_softmax(-1); p = lp.exp()
    selected = (labels != -100).nonzero(as_tuple=True)[0]
    expected = torch.zeros_like(ce)
    expected[selected] = -lp[selected, labels[selected]]
    torch.testing.assert_close(ce, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(ent, -(p * lp).sum(-1), rtol=2e-5, atol=2e-5)
    assert torch.equal(pred, logits.argmax(-1))
    dg = p[selected]
    dg[torch.arange(selected.numel(), device='cuda'), labels[selected]] -= 1
    torch.testing.assert_close(grad[:selected.numel()], (dg * 128.0).half(), rtol=0.003, atol=2e-5)


def test_column_compute_weight_preserves_values_and_invalidates_after_update():
    from solution.online_head import column_weight
    weight = torch.randn(97, 64, device='cuda', dtype=torch.float16)
    value = column_weight(weight)
    torch.testing.assert_close(value, weight, rtol=0, atol=0)
    assert column_weight(weight) is value
    weight.mul_(1.1)
    updated = column_weight(weight)
    assert updated is not value
    torch.testing.assert_close(updated, weight, rtol=0, atol=0)
