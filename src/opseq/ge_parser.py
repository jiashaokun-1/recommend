"""GE 节点 dict 列表 → ModelOpSeqCapture（纯逻辑，无硬件依赖）。"""

from __future__ import annotations

from opseq.ir import (
    DeviceInfo, InputSpec, ModelOpSeqCapture, OpNode, SourceMap, TensorDesc,
)
from opseq.op_normalizer import normalize_op_type

_GE_DTYPE_MAP = {
    "DT_FLOAT": "fp32",
    "DT_FLOAT16": "fp16",
    "DT_BF16": "bf16",
    "DT_INT8": "int8",
    "DT_UINT8": "uint8",
    "DT_INT32": "int32",
    "DT_INT64": "int64",
    "DT_BOOL": "bool",
}

# GE 图里的结构性节点，不是计算算子，跳过
_SKIP_TYPES = {"Data", "Const", "Constant", "Variable", "NetOutput"}


def normalize_dtype(ge_dtype: str) -> str:
    return _GE_DTYPE_MAP.get(ge_dtype, ge_dtype.lower())


def _tensor_from_desc(d: dict) -> TensorDesc:
    return TensorDesc(
        shape=list(d.get("shape", [])),
        dtype=normalize_dtype(d.get("dtype", "")),
        format=d.get("format", "ND"),
        stride=d.get("stride"),
    )


def parse_ge_graph(
    nodes: list[dict],
    *,
    model_id: str,
    model_name: str,
    chip: str,
    input_spec: InputSpec,
    backend: str = "ascend",
    skip_types: set[str] | None = None,
) -> ModelOpSeqCapture:
    skip = _SKIP_TYPES if skip_types is None else skip_types
    ops: list[OpNode] = []
    next_id = 0
    for node in nodes:
        if node.get("type") in skip:
            continue
        attr = dict(node.get("attr", {}))
        fusion_group = int(attr.get("fusion_group", next_id))
        ops.append(
            OpNode(
                id=next_id,
                op_type=normalize_op_type(node.get("type", "")),
                backend_op_name=node.get("name", ""),
                fusion_group_id=fusion_group,
                inputs=[_tensor_from_desc(x) for x in node.get("input_desc", [])],
                outputs=[_tensor_from_desc(x) for x in node.get("output_desc", [])],
                attrs=attr,
                source_map=SourceMap(
                    module_path=attr.get("_module_path", ""),
                    aten_op=attr.get("_aten_op", ""),
                ),
                measured=None,
            )
        )
        next_id += 1

    capture = ModelOpSeqCapture(
        model_id=model_id,
        model_name=model_name,
        backend=backend,
        device_info=DeviceInfo(chip=chip),
        input_spec=input_spec,
        ops=ops,
    )
    capture.validate()
    return capture
