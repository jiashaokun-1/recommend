import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchrec")

from opseq.input_builder import ShapeConfig, SparseFeatureConfig, build_inputs


def test_dense_shape():
    cfg = ShapeConfig(
        batch_size=1024, dense_dim=256,
        sparse=SparseFeatureConfig(keys=["user", "item"],
                                   pooling_factors={"user": 3, "item": 5}),
        dtype="float16",
    )
    out = build_inputs(cfg)
    assert tuple(out["dense"].shape) == (1024, 256)
    assert out["dense"].dtype == torch.float16


def test_sparse_kjt_lengths_and_values():
    cfg = ShapeConfig(
        batch_size=4, dense_dim=8,
        sparse=SparseFeatureConfig(keys=["user", "item"],
                                   pooling_factors={"user": 3, "item": 5}),
        dtype="float16",
    )
    out = build_inputs(cfg)
    kjt = out["sparse"]
    assert kjt.lengths().numel() == 2 * 4
    assert kjt.values().numel() == 4 * (3 + 5)
    assert kjt.keys() == ["user", "item"]
