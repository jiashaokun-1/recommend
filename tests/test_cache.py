from opseq.cache import IRCache, shape_hash
from opseq.ir import (
    ModelOpSeqCapture, DeviceInfo, InputSpec, InputDesc,
)


def _cap():
    return ModelOpSeqCapture(
        model_id="dlrm_v1", model_name="DLRM", backend="ascend",
        device_info=DeviceInfo(chip="Ascend910B"),
        input_spec=InputSpec(inputs=[InputDesc(name="dense", shape=[1024, 256], dtype="fp16")]),
    )


def test_shape_hash_stable_and_shape_sensitive():
    a = InputSpec(inputs=[InputDesc(name="x", shape=[1024, 256], dtype="fp16")])
    b = InputSpec(inputs=[InputDesc(name="x", shape=[2048, 256], dtype="fp16")])
    assert shape_hash(a) == shape_hash(a)
    assert shape_hash(a) != shape_hash(b)


def test_put_then_get_round_trip(tmp_path):
    cache = IRCache(root=tmp_path)
    cap = _cap()
    h = shape_hash(cap.input_spec)
    cache.put(cap, shape_hash=h)
    got = cache.get(model_id="dlrm_v1", shape_hash=h, backend="ascend")
    assert got == cap


def test_get_missing_returns_none(tmp_path):
    cache = IRCache(root=tmp_path)
    assert cache.get(model_id="nope", shape_hash="x", backend="ascend") is None
