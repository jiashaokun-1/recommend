from pathlib import Path

from opseq.profiler_attach import attach_measured, parse_profiler
from opseq.ir import (
    ModelOpSeqCapture, OpNode, TensorDesc, DeviceInfo, InputSpec, InputDesc,
)

FIX = Path(__file__).parent / "fixtures" / "sample_profiler.csv"


def test_parse_profiler():
    lat = parse_profiler(FIX)
    assert lat["MatMulV2_0"] == 12.5
    assert lat["Relu_0"] == 1.2


def _cap():
    return ModelOpSeqCapture(
        model_id="m", model_name="M", backend="ascend",
        device_info=DeviceInfo(chip="Ascend910B"),
        input_spec=InputSpec(inputs=[InputDesc(name="d", shape=[1, 1], dtype="fp16")]),
        ops=[
            OpNode(id=0, op_type="MatMul", backend_op_name="MatMulV2_0", fusion_group_id=0,
                   outputs=[TensorDesc(shape=[1, 1], dtype="fp16")]),
            OpNode(id=1, op_type="Unknown", backend_op_name="Missing_1", fusion_group_id=1),
        ],
    )


def test_attach_fills_measured_by_name():
    cap = _cap()
    out = attach_measured(cap, {"MatMulV2_0": 12.5})
    assert out.ops[0].measured.latency_us == 12.5
    assert out.ops[1].measured is None
    assert cap.ops[0].measured is None
