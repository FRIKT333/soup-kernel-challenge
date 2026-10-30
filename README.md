# Soup Kernel Challenge: High-Performance GPU Kernel Solution

> **Decentrathon 6.0 | Track: Soup Kernel Challenge**  
> **Team:** MT Kulsary (Roman Kravshenko / [FRIKT333](https://github.com/FRIKT333))  
> **Target:** NVIDIA Tesla T4 GPU (Turing `sm_75`, 15 GB VRAM)  
> **Public Repository:** [https://github.com/FRIKT333/soup-kernel-challenge](https://github.com/FRIKT333/soup-kernel-challenge)  
> **Public Kaggle Notebook:** [https://www.kaggle.com/code/romankravshenko/soup-kernel-challenge-t4](https://www.kaggle.com/code/romankravshenko/soup-kernel-challenge-t4)

---

## 1. What We Did (Architecture & Kernel Design)

Our goal is to maximize training speedup ($s = \text{base\_time} / \text{sol\_time}$, weight 35 pts) and memory reduction ratio ($m = \text{base\_peak} / \text{sol\_peak}$, weight 30 pts) for fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` on a single Tesla T4 GPU under `soup-cli[train]==0.77.0`, strictly passing all 10 automatic checks of `harness/check_rules.py`.

### 1.1 Bottleneck Analysis

Profiling base Soup with `Qwen2.5-1.5B` (NF4, LoRA $r=16, \alpha=32$ on `q_proj`/`v_proj`, batch 4, seq_len 512, $V=151,936$ vocab) revealed two critical bottlenecks:

1. **Peak Memory Bottleneck (Output Head & Logits):**
   - The output head maps hidden states $[4 \times 512, 1536]$ to full logits $[2048, 151936]$.
   - In fp16, the logits tensor alone is $622\text{ MB}$. During standard PyTorch `CrossEntropyLoss` and metrics calculation (`entropy_from_logits`, `argmax`, `row_losses`), multiple fp32 buffers and backward gradient tensors are materialized simultaneously, driving peak VRAM to $>14,100\text{ MiB}$ (near OOM on T4).
2. **Compute & Kernel Launch Overhead (Normalization Layers):**
   - Qwen2.5-1.5B has 28 transformer layers, each containing 2 `Qwen2RMSNorm` layers plus a final norm ($57$ norms forward $+ 57$ backward $= 114$ normalization passes per step).
   - In standard PyTorch, each RMSNorm executes 4–5 individual kernels (`pow`, `mean`, `rsqrt`, `mul`, type cast), generating $>500$ tiny kernel launches per step with repeated global memory round-trips.

---

### 1.2 Our Triton Kernels

We implemented custom `@triton.jit` kernels in `solution/`:

#### A. Fused Chunked Cross-Entropy + Online Metrics (`solution/cross_entropy.py`)
- **Kernel:** `fused_chunked_ce_kernel`
- **Mechanism:**
  - Evaluates the output head in sequence chunks ($M = 256$ tokens). Logits are materialized only for the current chunk ($[256, 151936]$, $<78\text{ MB}$), slashing activation memory by over $2,000\text{ MiB}$.
  - **Online Softmax & Multi-Metric Fusion in Pass 1:** Across $V=151,936$ vocabulary elements (tiled in blocks of $4096$), a single kernel pass computes:
    - Log-Sum-Exp ($\text{LSE}$) with running max in `tl.float32`.
    - Online Shannon Entropy: $\mathcal{H} = \text{LSE} - \frac{\sum z_i \exp(z_i - m)}{\sum \exp(z_i - m)}$.
    - Online Argmax: exact token prediction for `mean_token_accuracy`.
    - Per-token loss: $\mathcal{L}_i = \text{LSE} - z_{\text{target}}$.
  - **In-Place Analytical Gradient in Pass 2:** Overwrites the chunk logits buffer directly with:
    $$d\text{logits}_{i, k} = \frac{p_{i, k} - \mathbf{1}_{k = y_i}}{\text{denominator}}$$
  - Computes hidden state gradient via $dH_{\text{chunk}} = d\text{logits} \times W_{\text{head}}$, requiring **zero** PyTorch autograd graph buffers for the 152k vocabulary.
  - Generates exact `token_loss` for `rewind.jsonl` row losses and `_log_trl_metrics`.

#### B. Fused Qwen2RMSNorm Forward & Backward (`solution/rmsnorm.py`)
- **Kernels:** `fused_rmsnorm_fwd_kernel` & `fused_rmsnorm_bwd_kernel`
- **Mechanism:**
  - For hidden dimension $D = 1536$, launches with `BLOCK_SIZE = 2048` and `num_warps = 8`.
  - **Forward:** Loads fp16 activations, computes mean-square and reciprocal square root ($r = (s + \epsilon)^{-1/2}$) in registers in `tl.float32`, scales by weights, and writes fp16 outputs directly. Saves $r$ ($[M]$ scalar tensor) for backward.
  - **Backward:** Evaluates exact analytical gradient in a single kernel launch:
    $$dx_i = r \cdot \left[ dy_i w_i - \frac{r^2}{D} x_i \cdot \sum_j (dy_j w_j x_j) \right]$$
  - Replaces hundreds of elementwise PyTorch launches with 2 fast fused Triton calls per layer.

#### C. Preserved SwiGLU Kernel (`solution/swiglu.py`)
- Retains the fused SwiGLU kernel to eliminate intermediate gate/up activation buffers across all 28 MLP blocks, preserving the baseline's ~2,000 MiB saving.

---

## 2. How to Run It (Reproducibility)

### 2.1 Kaggle T4 Environment Setup
1. Create a Kaggle notebook with:
   - **Accelerator:** GPU T4 x2 (Harness uses GPU 0).
   - **Internet:** On.
2. In the first code cell, set repository and commit:
   ```python
   REPO_URL = "https://github.com/FRIKT333/soup-kernel-challenge.git"
   COMMIT = "08e655320cc82347699802aab3356b96c4b56f65"
   ```
3. Run the notebook (`Run All` / `Save & Run All`). The notebook executes:
   ```bash
   # 1. Download and format public training & eval data
   python task/prepare_data.py

   # 2. Run kernel unit tests
   python -m pytest solution -q

   # 3. Run complete benchmark (warmup, base soup, solution soup, adapter eval)
   python harness/run_task.py

   # 4. Run automated anti-cheat & regression rules check
   python harness/check_rules.py

   # 5. Compute official competition score
   python harness/score.py runs/result.json
   ```

### 2.2 Local Verification Commands
To verify correctness and rules locally:
```bash
# Verify AST anti-cheat and kernel registry
python -c "import sys; sys.path.insert(0, 'harness'); from check_rules import source_kernels; print(source_kernels('solution'))"

# Verify that harness, tests, and task are 100% untouched
git diff harness/ tests/ task/

# Run Soup 79 regression tests with our solution applied
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

---

## 3. Results & Compliance

### 3.1 Benchmark Comparison (Kaggle Tesla T4)

| Metric | Base Soup (`0.77.0`) | Solution (Ours) | Gain / Delta |
|---|---|---|---|
| **Wall Time (`soup train`)** | ~142.5 s | ~118.0 s | **~1.21x Speedup ($s$)** |
| **Peak GPU Memory (`nvidia-smi`)** | ~14,120 MiB | ~9,450 MiB | **~1.49x Memory Ratio ($m$)** |
| **Quality (Held-Out Eval Loss)** | 2.148 | 2.149 | **+0.05% (Limit: $\le +0.5\%$)** |
| **Own Triton Kernel Share (Step 3+)** | 0.0% | **~11.5%** | **PASS (Limit: $\ge 5.0\%$)** |

### 3.2 Rule Checks (`harness/check_rules.py`)

All 10 checks pass cleanly:
1. `PASS` — Every base and solution run finished with exit code 0.
2. `PASS` — No ready-made kernel libraries imported or loaded.
3. `PASS` — Own `@triton.jit` kernels in `solution/` launched in training (`fused_chunked_ce_kernel`, `fused_rmsnorm_fwd_kernel`, `fused_rmsnorm_bwd_kernel`).
4. `PASS` — Own kernels together take $\ge 5\%$ of training time ($11.5\%$).
5. `PASS` — Same task and Soup outputs: steps, adapter tensors, settings, tokens per step, rewind rows, entropy, accuracy, and rewind row losses within 3%.
6. `PASS` — Loss curve follows the base: step 0 within 1%, mean of all steps within 3%.
7. `PASS` — Quality: eval loss at most 0.5% above the base (+0.05%).
8. `PASS` — Not worse than base on time or memory.
9. `PASS` — Clean measurement on Tesla T4.
10. `PASS` — All 79 Soup regression tests pass with patches applied.

---

## 4. Limitations & Future Work

1. **Static Block Sizes:**
   - `fused_rmsnorm` is compiled with `BLOCK_SIZE = 2048`, designed for $D \le 2048$ ($D=1536$ for Qwen2.5-1.5B). Larger models ($D > 2048$) would require multiple blocks or dynamic block dispatch.
2. **Vocabulary Tiling:**
   - `fused_chunked_ce_kernel` tiles vocabulary in blocks of 4096. On architectures with different shared memory sizes (e.g. Ampere/Hopper), tiling could be adjusted to 8192 for even higher memory bandwidth utilization.
3. **Future Optimizations:**
   - Fusing Rotary Position Embeddings (`apply_rotary_pos_emb`) with Q/K projection kernels.
   - Pipelined double-buffering of hidden state chunks to overlap GEMM compute with Triton reduction.
