from opseq.ir import (
    ModelOpSeqCapture, OpNode, TensorDesc, InputSpec, InputDesc,
    DeviceInfo, SourceMap, Measured,
)


def _sample_capture() -> ModelOpSeqCapture:
    return ModelOpSeqCapture(
        model_id="dlrm_v1",
        model_name="DLRM",
        backend="ascend",
        device_info=DeviceInfo(chip="Ascend910B"),
        input_spec=InputSpec(
            inputs=[InputDesc(name="dense", shape=[1024, 256], dtype="fp16", format="ND")],
            symbol_bindings={},
        ),
        ops=[
            OpNode(
                id=0,
                op_type="MatMul",
                backend_op_name="MatMulV2_0",
                fusion_group_id=0,
                inputs=[TensorDesc(shape=[1024, 256], dtype="fp16", format="ND")],
                outputs=[TensorDesc(shape=[1024, 128], dtype="fp16", format="ND")],
                attrs={"transpose_x2": False},
                source_map=SourceMap(module_path="mlp.0", aten_op="aten::addmm"),
                measured=None,
            )
        ],
    )


def test_round_trip_json():
    cap = _sample_capture()
    restored = ModelOpSeqCapture.from_dict(cap.to_dict())
    assert restored == cap


def test_symbolic_dim_allowed():
    td = TensorDesc(shape=[1024, "s0"], dtype="fp16")
    assert td.shape == [1024, "s0"]


def test_validate_rejects_duplicate_ids():
    cap = _sample_capture()
    cap.ops.append(cap.ops[0])  # 重复 id=0
    import pytest
    with pytest.raises(ValueError, match="duplicate op id"):
        cap.validate()


def test_measured_round_trip():
    cap = _sample_capture()
    cap.ops[0].measured = Measured(latency_us=12.5)
    restored = ModelOpSeqCapture.from_dict(cap.to_dict())
    assert restored.ops[0].measured == Measured(latency_us=12.5)
