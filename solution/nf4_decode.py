"""Independent fused double-quantized NF4 -> FP16 decoder.

Quantization remains bitsandbytes NF4. This combines the two scale-decode
passes, offset addition and nibble decoding; no pretrained values are replaced.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["SIZE"])
def nf4_double_decode_kernel(PACKED, SCALES, CODE8, MAX2, OFFSET, CODE4, OUT,
                             SIZE, GROUP: tl.constexpr, GROUP2: tl.constexpr,
                             BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    group = i // GROUP
    qscale = tl.load(SCALES + group, i < SIZE, 0).to(tl.int32)
    normalized = tl.load(CODE8 + qscale)
    maximum = tl.load(MAX2 + group // GROUP2, i < SIZE, 0)
    offset = tl.load(OFFSET)
    scale = normalized * maximum + offset
    byte = tl.load(PACKED + i // 2, i < SIZE, 0).to(tl.int32)
    code = tl.where(i % 2 == 0, byte >> 4, byte & 15)
    value = tl.load(CODE4 + code)*scale
    tl.store(OUT + i, value, i < SIZE)


def supported(state):
    return (state is not None and state.quant_type == "nf4" and state.nested
            and state.dtype == torch.float16 and state.state2.dtype == torch.float32
            and state.blocksize == 64 and state.state2.blocksize == 256)


def decode_nf4(packed, state):
    size = int(state.shape[0])*int(state.shape[1])
    result = torch.empty(tuple(state.shape), device=packed.device, dtype=torch.float16)
    nf4_double_decode_kernel[(triton.cdiv(size, 1024),)](
        packed, state.absmax, state.state2.code, state.state2.absmax,
        state.offset, state.code, result, size, state.blocksize, state.state2.blocksize,
        1024, num_warps=4, enable_fp_fusion=False)
    return result


@triton.jit
def nf4_mlp_decode_kernel(P0, S0, C0, M0, O0, V0,
                          P1, S1, C1, M1, O1, V1,
                          P2, S2, C2, M2, O2, V2, OUT,
                          SIZE: tl.constexpr, BLOCK: tl.constexpr):
    matrix = tl.program_id(1)
    if matrix == 0:
        p, s, c, m, o, v = P0, S0, C0, M0, O0, V0
    elif matrix == 1:
        p, s, c, m, o, v = P1, S1, C1, M1, O1, V1
    else:
        p, s, c, m, o, v = P2, S2, C2, M2, O2, V2
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group = i // 64
    qscale = tl.load(s + group, i < SIZE, 0).to(tl.int32)
    scale = tl.load(c + qscale) * tl.load(m + group // 256, i < SIZE, 0) + tl.load(o)
    byte = tl.load(p + i // 2, i < SIZE, 0).to(tl.int32)
    code = tl.where(i % 2 == 0, byte >> 4, byte & 15)
    tl.store(OUT + matrix * SIZE + i, tl.load(v + code) * scale, i < SIZE)


def decode_mlp(weights, states):
    """Decode three equal-sized matrices in one launch, retaining their states."""
    size = int(states[0].shape[0]) * int(states[0].shape[1])
    out = torch.empty(3 * size, device=weights[0].device, dtype=torch.float16)
    args = []
    for weight, state in zip(weights, states):
        args.extend((weight, state.absmax, state.state2.code,
                     state.state2.absmax, state.offset, state.code))
    nf4_mlp_decode_kernel[(triton.cdiv(size, 1024), 3)](
        *args, out, size, 1024, num_warps=4, enable_fp_fusion=False)
    return out

