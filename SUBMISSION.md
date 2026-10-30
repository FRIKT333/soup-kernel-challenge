# Submission: MT Kulsary / FRIKT333

## Links
- **Repository, final commit:** https://github.com/FRIKT333/soup-kernel-challenge/commit/08e655320cc82347699802aab3356b96c4b56f65
- **Official Hackathon Private Repo:** https://github.com/BAITC-Hacks/hack-81eedec3-mtkulsary/commit/08e655320cc82347699802aab3356b96c4b56f65
- **Public Kaggle notebook:** https://www.kaggle.com/code/romankravshenko/soup-kernel-challenge-t4
- **Demo video (up to 3 minutes):** [Link to Video]
- **Presentation:** [Link to Presentation]

## Team (MT Kulsary)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Roman Kravshenko | GPU Kernel & Architecture Engineer | [FRIKT333](https://github.com/FRIKT333) | [romankravshenko](https://www.kaggle.com/romankravshenko) | romankravshenko7@gmail.com |

## Our Kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of training time (check 4) |
|---|---|---|---|
| `fused_chunked_ce_kernel` | `solution/cross_entropy.py` | Full-vocabulary Cross-Entropy forward & backward, online Shannon entropy, argmax token accuracy | ~6.5% - 8.2% |
| `fused_rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | PyTorch Qwen2RMSNorm forward (pow + mean + rsqrt + mul + cast) | ~2.1% - 3.0% |
| `fused_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | PyTorch Qwen2RMSNorm backward autograd graph | ~2.3% - 3.2% |

### Total Own Triton Kernel Time Share
- **Combined Share:** $\ge 10.5\% - 14.4\%$ of step 3+ training time (exceeds the 5.0% threshold required by check 4).
- **Novelty:** 100% custom AST, 0.0% likeness to `example_reference.py`, zero banned third-party libraries (`unsloth`, `liger_kernel`, `flash_attn`, `xformers` etc.).

### Ideas & References
- Online Softmax reduction algorithm: Milakov & Gimelshteyn (2018).
- Exact RMSNorm backward analytical gradient derivation in registers.
- Chunked loss streaming architecture compatible with `soup-cli==0.77.0`.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json`)
| Metric | Base Soup | Solution (Ours) | Ratio / Delta |
|---|---|---|---|
| Time of `soup train`, s | ~142.5 s | ~118.0 s | **~1.21x speedup ($s$)** |
| Peak GPU memory, MiB | ~14,120 MiB | ~9,450 MiB | **~1.49x memory ratio ($m$)** |
| Eval loss (held-out rows) | 2.148 | 2.149 | **+0.05% vs base ($\le 0.5\%$)** |

- [x] All 10 checks of `harness/check_rules.py` pass (`PASS`).
- [x] `README.md` rewritten for our solution: what we did, how to run it, results, limits.
- [x] All 79 Soup regression tests pass with patches applied (`SOUP_APPLY_SOLUTION=1`).
- [x] Exact output parity: `rewind.jsonl` row losses, `entropy`, and `mean_token_accuracy` match within 3%.

## Presentation Summary
1. **Bottleneck Analysis:**
   - 152k vocabulary logits materialization consumes over 2.4 GB of VRAM.
   - 114 normalization passes per step launch over 500 elementwise PyTorch kernels.
2. **Kernel Engineering:**
   - Fused chunked cross-entropy executes online softmax, LSE, entropy, argmax accuracy, and analytical gradient calculation in Triton in chunks of 64 tokens.
   - Fused RMSNorm forward and backward with float32 accumulators in registers (`BLOCK_SIZE=2048`).
3. **Verification:**
   - Unit tests against PyTorch reference with strict tolerances.
   - Full compatibility with Soup rewind logger and evaluation scripts.

## Limitations
- Block size of 2048 in RMSNorm is optimized for hidden dimension $D \le 2048$ (Qwen2.5-1.5B has $D=1536$).
- Cross-entropy tile size 4096 is tuned for memory bandwidth on Turing T4 (`sm_75`).
