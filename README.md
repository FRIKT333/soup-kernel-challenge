# Soup Kernel Challenge: High-Performance GPU Kernel Solution

> **Decentrathon 6.0 | Track: Soup Kernel Challenge**  
> **Team:** MT Kulsary (Roman Kravshenko / [FRIKT333](https://github.com/FRIKT333))  
> **Target:** NVIDIA Tesla T4 GPU (Turing `sm_75`, 15 GB VRAM)  
> **Branch:** `teamwork/t4-45s-speedup`  
> **Public Repository:** [https://github.com/FRIKT333/soup-kernel-challenge](https://github.com/FRIKT333/soup-kernel-challenge)  
> **Public Kaggle Notebook:** [https://www.kaggle.com/code/frikt333/soup-t4-final-confirmation](https://www.kaggle.com/code/frikt333/soup-t4-final-confirmation)

---

## 1. What We Did (Architecture & Kernel Design)

Our goal is to maximize training speedup ($s = \text{base\_time} / \text{sol\_time}$, weight 35 pts) and memory reduction ratio ($m = \text{base\_peak} / \text{sol\_peak}$, weight 30 pts) for fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` on a single Tesla T4 GPU under `soup-cli[train]==0.77.0`, strictly passing all 10 automatic checks of `harness/check_rules.py`.

### 1.1 Bottleneck Analysis

Profiling base Soup with `Qwen2.5-1.5B` (NF4, LoRA $r=16, \alpha=32$ on `q_proj`/`v_proj`, batch 4, seq_len 512, $V=151,936$ vocab) revealed three critical bottlenecks:

1. **Peak Memory Bottleneck (Output Head & Logits):**
   - The output head maps hidden states $[4 \times 512, 1536]$ to full logits $[2048, 151936]$.
   - In fp16, the logits tensor alone is $622\text{ MB}$. During standard PyTorch `CrossEntropyLoss` and metrics calculation (`entropy_from_logits`, `argmax`, `row_losses`), multiple fp32 buffers and backward gradient tensors are materialized simultaneously, driving peak VRAM to $>14,300\text{ MiB}$ (near OOM on T4).
2. **Double-Dequantization Overhead (Frozen NF4 Weights):**
   - Decoding 112 attention linear layers and 28 MLP blocks from NF4 double-quantization on every forward and backward pass saturates memory bandwidth and introduces thousands of tiny kernel launches.
3. **Cold JIT Compilation Stalls:**
   - Under standard execution, Triton JIT compiles multiple kernel variants on step 0, stalling training by up to 7–8 seconds.

---

### 1.2 Our Triton Kernels & Architecture

We implemented a coordinated suite of custom `@triton.jit` kernels in `solution/`:

#### A. Online Vocabulary Gradient Head (`solution/online_head.py`)
- **Kernel:** `online_vocabulary_gradient_kernel`
- **Mechanism:**
  - Evaluates the output head in token chunks ($M = 256$ tokens) using cuBLAS GEMM, keeping logits transient in $O(\text{chunk} \times V)$ memory ($<78\text{ MB}$).
  - In a single streaming kernel pass, calculates Log-Sum-Exp ($\text{LSE}$), online Shannon entropy, argmax prediction, and supervised CE loss.
  - Computes and immediately projects backward hidden gradients $dx = d\text{logits} \cdot W_{\text{head}}$ during the forward pass, reducing backward memory to $O(1)$ and eliminating all backward classifier GEMMs.

#### B. Vectorized NF4 MLP Dequantization (`solution/nf4_decode.py`)
- **Kernels:** `nf4_mlp_decode_kernel` & `nf4_double_decode_kernel`
- **Mechanism:**
  - High-throughput vectorized decoding of NF4 weights with double-quantization scale lookup directly into FP16.
  - Operates on all uncached MLP layers across the 28 blocks, ensuring our own Triton kernels account for a solid **12.4%** of GPU training time (comfortably exceeding the $\ge 5.0\%$ rule).

#### C. Compact Rotary Position Embeddings (`solution/attention.py`)
- **Kernel:** `packed_rotary_kernel`
- **Mechanism:**
  - Applies RoPE rotations directly on packed sequence tokens for query ($H=12$) and key ($H=2$) heads with exact FP16 product rounding parity.
  - Uses `do_not_specialize=['TOTAL']` to avoid JIT recompilation across varying batch lengths.

#### D. Fused Qwen2RMSNorm Forward & Backward (`solution/rmsnorm.py`)
- **Kernels:** `frozen_rms_forward_kernel` & `frozen_rms_backward_kernel`
- **Mechanism:**
  - Forward computes variance and rsqrt in FP32 accumulators (`BLOCK_SIZE = 2048`, `num_warps = 4`) and writes normalized FP16 activations with optional packed token mapping.
  - Backward computes analytical gradients with exact combined FP16 reduction rounding.

#### E. Rank-Boundary LoRA Fusion (`solution/lora_kernels.py`)
- **Kernels:** `lora_up_add_kernel` & `lora_cast_scale_gradient_kernel`
- **Mechanism:**
  - Fuses the rank-16 addition $h \cdot B + \text{base}$ with in-register scaling and conversion, eliminating temporary tensor allocations.

#### F. Cold JIT Compilation Overlap (`solution/cold_compile.py`)
- Pre-compiles all representative kernel signatures concurrently across 4 CPU threads during model weight loading.
- Eliminates step-0 JIT stalls completely, dropping cold start wall time from 57.8s to 53.3s.

---

## 2. Reproduction Guide

### 2.1 Execution on Kaggle (Tesla T4)
```bash
# 1. Clone repository
git clone --branch teamwork/t4-45s-speedup https://github.com/FRIKT333/soup-kernel-challenge.git /kaggle/working/repo
cd /kaggle/working/repo

# 2. Prepare data
python task/prepare_data.py

# 3. Run full ABBA comparative benchmark
python harness/run_task.py --repeats 4

# 4. Run automated anti-cheat & regression rules check
python harness/check_rules.py

# 5. Compute official competition score
python harness/score.py runs/result.json
```

### 2.2 Local Verification Commands
```bash
# Run unit tests on solution kernels
python -m pytest solution -q

# Run Soup regression tests with solution applied
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q

# Check rules and AST anti-cheat compliance
python harness/check_rules.py --skip-tests
```

---

## 3. Results & Compliance

### 3.1 Official ABBA Benchmark (Kaggle Tesla T4, 4 Repeats)

| Metric | Base Soup (`0.77.0`) | Solution (Ours) | Gain / Delta |
|---|---|---|---|
| **Wall Time (`soup train`)** | 82.23 s | **53.53 s** (52.8s warm) | **1.536x Wall Speedup ($s$)** |
| **Training Loop Time (64 steps)** | 60.24 s | **30.73 s** (25.3s timed loop) | **1.96x Loop Speedup** |
| **Peak GPU Memory (`nvidia-smi`)** | 14,345 MiB | **4,409 MiB** | **3.25x Memory Ratio ($m$)** (70% savings) |
| **Quality (Held-Out Eval Loss)** | 2.04964 | **2.04967** | **+0.001% (Limit: $\le +0.5\%$)** |
| **Loss Curve Tracking Delta** | - | **0.01%** | **PASS (Limit: $\le 3.0\%$)** |
| **Own Triton Kernel Share (Step 3+)** | 0.0% | **12.4%** | **PASS (Limit: $\ge 5.0\%$)** |

### 3.2 Rule Checks (`harness/check_rules.py`)

All 10 checks pass cleanly (`10 / 10 PASS`):
1. `PASS` — Every base and solution run finished with exit code 0.
2. `PASS` — No ready-made kernel libraries imported or loaded.
3. `PASS` — Own `@triton.jit` kernels in `solution/` launched in training.
4. `PASS` — Own kernels together take $\ge 5\%$ of training time (**12.4%**).
5. `PASS` — Same task and Soup outputs: steps, adapter tensors, settings, tokens per step, rewind rows, entropy, accuracy, and rewind row losses within 3%.
6. `PASS` — Loss curve follows the base: step 0 within 1%, mean of all steps within 3%.
7. `PASS` — Quality: eval loss at most 0.5% above the base (**+0.001%**).
8. `PASS` — Not worse than base on time or memory.
9. `PASS` — Clean measurement on Tesla T4.
10. `PASS` — All 79 Soup regression tests pass with patches applied.

---

## 4. Limitations & Future Work

1. **Head Specialization:**
   - Rotary kernel block layouts are specialized for Qwen2.5-1.5B head configurations ($D=128$, $H=12$, $KV=2$).
2. **RMSNorm Block Size:**
   - `frozen_rms_forward_kernel` is compiled with `BLOCK_SIZE = 2048`, designed for $D \le 2048$.
3. **Platform Tuning:**
   - Tailored and verified for NVIDIA Tesla T4 GPU (`sm_75`, Turing architecture).
