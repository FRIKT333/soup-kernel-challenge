"""Shape-only workspace selection for the streamed frozen classifier."""

HEAD_WORKSPACE_BYTES = 288 * 1024 * 1024

def choose_head_chunk(vocabulary):
    rows = max(16, HEAD_WORKSPACE_BYTES // (4 * vocabulary))
    return min(256, max(16, rows // 16 * 16))


NF4_CACHE_BYTES = 310 * 1024 * 1024


def choose_nf4_cache(model, budget_bytes=None):
    """Keep small decoded matrices; other values are restored from exact NF4.

    Decisions use only public weight shapes and a fixed byte budget, independent
    of data, timing, free memory or compilation state. Quantized weights survive.
    """
    import bitsandbytes as bnb
    import torch
    budget = NF4_CACHE_BYTES if budget_bytes is None else int(budget_bytes)
    candidates, seen = [], set()
    for name, module in model.named_modules():
        if type(module) is not bnb.nn.Linear4bit or module.weight.requires_grad:
            continue
        state = module.weight.quant_state
        if "mlp" in name:
            if state is not None:
                state._permit_frozen_cache = False
                if hasattr(state, "_frozen_dense_cache"):
                    del state._frozen_dense_cache
            continue
        if (state is not None and state.quant_type == "nf4" and state.dtype == torch.float16
                and id(state) not in seen):
            seen.add(id(state))
            size = int(state.shape[0]) * int(state.shape[1]) * 2
            candidates.append((size, name, state))
    used = 0
    for size, name, state in sorted(candidates, key=lambda item: (item[0],item[1])):
        state._permit_frozen_cache = used + size <= budget
        if state._permit_frozen_cache:
            used += size
        elif hasattr(state, "_frozen_dense_cache"):
            del state._frozen_dense_cache
    return used
