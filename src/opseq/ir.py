from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union

Dim = Union[int, str]  # 现在存 int；未来塞符号表达式字符串（如 "s0*256"）


@dataclass
class TensorDesc:
    shape: list[Dim]
    dtype: str
    format: str = "ND"
    stride: Optional[list[int]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TensorDesc":
        return cls(shape=list(d["shape"]), dtype=d["dtype"],
                   format=d.get("format", "ND"), stride=d.get("stride"))


@dataclass
class SourceMap:
    module_path: str = ""
    aten_op: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "SourceMap":
        return cls(module_path=d.get("module_path", ""), aten_op=d.get("aten_op", ""))


@dataclass
class Measured:
    latency_us: float

    @classmethod
    def from_dict(cls, d: dict) -> "Measured":
        return cls(latency_us=d["latency_us"])


@dataclass
class OpNode:
    id: int
    op_type: str
    backend_op_name: str
    fusion_group_id: int
    inputs: list[TensorDesc] = field(default_factory=list)
    outputs: list[TensorDesc] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    source_map: SourceMap = field(default_factory=SourceMap)
    measured: Optional[Measured] = None

    @classmethod
    def from_dict(cls, d: dict) -> "OpNode":
        return cls(
            id=d["id"],
            op_type=d["op_type"],
            backend_op_name=d["backend_op_name"],
            fusion_group_id=d["fusion_group_id"],
            inputs=[TensorDesc.from_dict(x) for x in d.get("inputs", [])],
            outputs=[TensorDesc.from_dict(x) for x in d.get("outputs", [])],
            attrs=dict(d.get("attrs", {})),
            source_map=SourceMap.from_dict(d.get("source_map", {})),
            measured=Measured.from_dict(d["measured"]) if d.get("measured") else None,
        )


@dataclass
class InputDesc:
    name: str
    shape: list[Dim]
    dtype: str
    format: str = "ND"

    @classmethod
    def from_dict(cls, d: dict) -> "InputDesc":
        return cls(name=d["name"], shape=list(d["shape"]), dtype=d["dtype"],
                   format=d.get("format", "ND"))


@dataclass
class InputSpec:
    inputs: list[InputDesc] = field(default_factory=list)
    symbol_bindings: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "InputSpec":
        return cls(
            inputs=[InputDesc.from_dict(x) for x in d.get("inputs", [])],
            symbol_bindings=dict(d.get("symbol_bindings", {})),
        )


@dataclass
class DeviceInfo:
    chip: str

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceInfo":
        return cls(chip=d["chip"])


@dataclass
class ModelOpSeqCapture:
    model_id: str
    model_name: str
    backend: str
    device_info: DeviceInfo
    input_spec: InputSpec
    schema_version: str = "0.1"
    capture_mode: str = "specialize"
    guards: list[str] = field(default_factory=list)
    ops: list[OpNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelOpSeqCapture":
        return cls(
            model_id=d["model_id"],
            model_name=d["model_name"],
            backend=d["backend"],
            device_info=DeviceInfo.from_dict(d["device_info"]),
            input_spec=InputSpec.from_dict(d["input_spec"]),
            schema_version=d.get("schema_version", "0.1"),
            capture_mode=d.get("capture_mode", "specialize"),
            guards=list(d.get("guards", [])),
            ops=[OpNode.from_dict(x) for x in d.get("ops", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> "ModelOpSeqCapture":
        return cls.from_dict(json.loads(s))

    def validate(self) -> None:
        if self.backend not in ("ascend", "gpu"):
            raise ValueError(f"invalid backend: {self.backend}")
        seen = set()
        for op in self.ops:
            if op.id in seen:
                raise ValueError(f"duplicate op id: {op.id}")
            seen.add(op.id)
