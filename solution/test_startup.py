import torch


def test_fresh_map_identity_keeps_all_rows_order_values_and_rng():
    from datasets import Dataset
    from solution.startup import patch_preprocessing, _preparing

    patch_preprocessing()
    first_map = Dataset.map
    patch_preprocessing()
    assert Dataset.map is first_map
    data = Dataset.from_dict({"x": [3, 1, 4, 2]})
    calls = []

    def transform(row):
        calls.append(row["x"])
        return {"y": row["x"] * 7}

    rng = torch.get_rng_state()
    token = _preparing.set(True)
    try:
        a = data.map(transform)
        b = data.map(transform)
        explicit = data.map(transform, new_fingerprint="e" * 64)
    finally:
        _preparing.reset(token)
    expected = {"x": [3, 1, 4, 2], "y": [21, 7, 28, 14]}
    assert a.to_dict() == b.to_dict() == explicit.to_dict() == expected
    assert calls == [3, 1, 4, 2] * 3
    assert a._fingerprint != b._fingerprint
    assert explicit._fingerprint == "e" * 64
    assert torch.equal(rng, torch.get_rng_state())
