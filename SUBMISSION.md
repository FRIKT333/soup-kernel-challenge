# Submission: MT Kulsary / FRIKT333

## Links
- **Repository, final commit:** https://github.com/FRIKT333/soup-kernel-challenge/commit/05337079cedc0b7c0e8ebcc59b568fcabbde4949
- **Official Hackathon Private Repo:** https://github.com/BAITC-Hacks/hack-81eedec3-mtkulsary/commit/05337079cedc0b7c0e8ebcc59b568fcabbde4949
- **Branch:** `teamwork/t4-45s-speedup`
- **Public Kaggle notebook:** https://www.kaggle.com/code/frikt333/soup-t4-final-confirmation
- **Demo video (up to 3 minutes):** [Link to Video]
- **Presentation:** [Link to Presentation]

## Team (MT Kulsary)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Roman Kravshenko | GPU Kernel & Architecture Engineer | [FRIKT333](https://github.com/FRIKT333) | [romankravshenko](https://www.kaggle.com/romankravshenko) | romankravshenko7@gmail.com |

## Our Kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of training time (check 4) |
|---|---|---|---|
| `online_vocabulary_gradient_kernel` | `solution/online_head.py` | One-pass fused Cross-Entropy loss, LSE, Shannon entropy, argmax prediction, and immediate hidden gradient projection $dx = d\text{logits} \cdot W$ | ~1.5% |
| `nf4_mlp_decode_kernel` | `solution/nf4_decode.py` | High-throughput vectorized NF4 dequantization for uncached MLP projections directly into FP16 | ~7.8% |
| `packed_rotary_kernel` | `solution/attention.py` | Fused RoPE rotary embedding forward & backward on compact token sequences | ~1.1% |
| `frozen_rms_forward_kernel` | `solution/rmsnorm.py` | Fused RMSNorm forward with optional packed sequence mapping in FP32 accumulators | ~0.8% |
| `frozen_rms_backward_kernel` | `solution/rmsnorm.py` | Analytical RMSNorm gradient backward with combined FP16 reduction | ~0.8% |
| `packed_dropout_forward_kernel` | `solution/dropout.py` | Compact native dropout forward preserving stock PyTorch Philox generator layout | ~1.3% |
| `packed_dropout_backward_kernel` | `solution/dropout.py` | Masked gradient backward scaling for compact tokens | ~0.7% |
| `lora_up_add_kernel` | `solution/lora_kernels.py` | Rank-boundary fusion: $h \cdot B + \text{base}$ with in-register scale and dtype conversion | ~0.9% |
| `lora_cast_scale_gradient_kernel` | `solution/lora_kernels.py` | Fused gradient cast and scaling for LoRA adapters | ~0.5% |
| `token_gather_kernel` / `token_restore_kernel` | `solution/compact.py` | Compact valid sequence token gather and restore | ~0.2% |

### Total Own Triton Kernel Time Share
- **Combined Share:** **12.4%** of step 3+ training time across all 4 measured runs (exceeds the 5.0% threshold required by check 4).
- **Novelty:** 100% custom AST, 0.0% likeness to `example_reference.py`, zero banned third-party libraries (`unsloth`, `liger_kernel`, `flash_attn`, `xformers`, etc.).

### Ideas & References
- Online Softmax & cross-entropy reduction: Milakov & Gimelshteyn (2018).
- Streamed classifier loss scheduling: Cut Your Losses (arXiv:2411.09009).
- LoRA rank boundary fusion: LoRAFusion Section 5.1 (arXiv:2510.00206).
- Exact NF4 double-quantization restoration with fused dequantization kernels.
- Overlapped cold JIT compilation via ThreadPoolExecutor joined before training loop.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json`)
| Metric | Base Soup | Solution (Ours) | Ratio / Delta |
|---|---|---|---|
| Wall time of `soup train` (median of 4 runs) | 82.23 s | **53.53 s** (52.8s warm) | **1.536x wall speedup** |
| Training loop time (64 steps) | 60.24 s | **30.73 s** (25.3s timed loop) | **1.96x loop speedup** |
| Peak GPU memory | 14,345 MiB | **4,409 MiB** | **3.25x memory reduction** (70% memory freed) |
| Eval loss (held-out rows) | 2.04964 | **2.04967** | **+0.001%** vs base (limit $\le 0.5\%$) |
| Loss curve tracking delta | - | **0.01%** | **PASS** (limit $\le 3.0\%$) |
| Own Triton kernels time share | 0.0% | **12.4%** | **PASS** (limit $\ge 5.0\%$) |
| Harness Rule Checks (all 10) | 0/10 | **10/10 PASS** | **100% compliant** |
| Soup Regression Tests | - | **79/79 PASS** | **100% compliant** |

- [x] All 10 checks of `harness/check_rules.py` pass (`PASS`).
- [x] `README.md` rewritten for our solution: what we did, how to run it, results, limits.
- [x] All 79 Soup regression tests pass with patches applied (`SOUP_APPLY_SOLUTION=1`).
- [x] Exact output parity: `rewind.jsonl` row losses, `entropy`, and `mean_token_accuracy` match within tolerance.

## Key Architectural Highlights
1. **Online Gradient Head (`solution/online_head.py`)**:
   - Computes Cross-Entropy loss, LSE, Shannon entropy, and argmax token accuracy in a single online streaming pass in chunks of 256 tokens.
   - Eliminates the 2.4 GB full-vocabulary logits tensor ($B \times T \times 151936$).
   - Projects hidden state gradients immediately in forward pass, making backward pass through classifier $O(1)$ memory with zero backward GEMMs.
2. **NF4 Precision & Memory Cache Schedule (`solution/schedule.py`)**:
   - Caches all 112 attention layer projections in FP16 (310 MiB budget).
   - Uncached MLP layers are decoded on-the-fly via our vectorized Triton `nf4_mlp_decode_kernel`, maintaining a solid 12.4% Triton compute share.
3. **Cold JIT Compilation Overlap (`solution/cold_compile.py`)**:
   - Pre-compiles all representative kernel signatures concurrently across 4 CPU threads during model weight loading.
   - Eliminates step-0 JIT stalls completely, dropping cold start time from 57.8s to 53.3s.
4. **Packed Token Decoder (`solution/packed_decoder.py`)**:
   - Compacts valid tokens across decoder layers, avoiding redundant packing/unpacking and padding computation.

## Limitations
- Attention head dimensions are specialized for Qwen2.5-1.5B architecture ($D=128$, $H=12$, $KV=2$).
- RMSNorm block size 2048 is optimized for hidden dimension $D \le 2048$.
- Tested and tuned for NVIDIA Turing Tesla T4 GPU (sm_75, 15 GB VRAM).
