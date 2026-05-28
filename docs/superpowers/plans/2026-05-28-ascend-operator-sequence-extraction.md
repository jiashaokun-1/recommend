# 推荐模型 Kernel 级算子序抽取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 PyTorch 推荐模型静态抽取 Ascend GE 融合算子序（带 shape/dtype/format），产出后端中立的 JSON IR，并提供可选的 profiler 实测耗时回填工具。

**Architecture:** 流水线分两层——纯逻辑层（IR 数据结构、算子归一化、GE 图解析、缓存、profiler 解析与 join）全部可在无 NPU 环境下 TDD；硬件耦合层（torchair 编译+dump、profiler 运行）做薄封装，单测用 mock/fixture，真实验证在 Ascend 上集成。

**Tech Stack:** Python 3.10+，dataclasses + JSON（IR），pytest（测试），torch / torchrec（输入构造），torch_npu / torchair（Ascend 抽取，仅集成）。

---

## 文件结构

```
recommend/
  pyproject.toml                  # 项目+pytest 配置
  src/opseq/
    __init__.py
    ir.py                         # IR dataclasses + JSON 序列化/反序列化 + 校验
    op_normalizer.py              # GE 算子类型 → 归一化 op_type
    ge_parser.py                  # dtype 映射 + GE 节点 dict 列表 → ModelOpSeqCapture
    cache.py                      # 按 (model_id, shape_hash, backend) 存取 IR JSON
    input_builder.py              # 按 ShapeConfig 造 specialize 输入（含 KJT）
    ge_reader.py                  # 读 torchair dump 的 GE 图 → 节点 dict 列表（薄、格式耦合）
    ascend_extractor.py           # 编排：torchair 编译+dump → reader → parser → cache
    profiler_attach.py            # 可选：解析 profiler 输出 + 按名 join 回 IR
  tests/
    test_ir.py
    test_op_normalizer.py
    test_ge_parser.py
    test_cache.py
    test_input_builder.py
    test_ge_reader.py
    test_ascend_extractor.py
    test_profiler_attach.py
    fixtures/
      sample_ge_graph.json
      sample_profiler.csv
```

**模块边界**：`ir`/`op_normalizer`/`ge_parser`/`cache`/`profiler_attach` 为纯逻辑，零硬件依赖、全 TDD。`input_builder` 依赖 torch/torchrec（CPU 可测）。`ge_reader` 薄解析、fixture 测。`ascend_extractor` 编排、mock 测 + 设备集成。

---

## Task 0: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `src/opseq/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "opseq"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
    "npu: 需要 Ascend NPU 环境的集成测试",
    "torch: 需要 torch/torchrec 的测试",
]
```

- [ ] **Step 2: 创建包占位文件**

`src/opseq/__init__.py`:
```python
"""推荐模型 kernel 级算子序抽取。"""
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: 验证 pytest 可运行**

Run: `python -m pytest -q`
Expected: `no tests ran`（0 collected，无错误）

- [ ] **Step 4: 提交**

```bash
git add pyproject.toml src/opseq/__init__.py tests/__init__.py
git commit -m "chore: 初始化 opseq 包脚手架"
```

---

## Task 1: IR 数据结构与序列化

**Files:**
- Create: `src/opseq/ir.py`
- Test: `tests/test_ir.py`

- [ ] **Step 1: 写失败测试**

`tests/test_ir.py`:
```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_ir.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.ir'`

- [ ] **Step 3: 实现 ir.py**

`src/opseq/ir.py`:
```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_ir.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/opseq/ir.py tests/test_ir.py
git commit -m "feat: 公共 IR 数据结构与 JSON 序列化"
```

---

## Task 2: 算子类型归一化

**Files:**
- Create: `src/opseq/op_normalizer.py`
- Test: `tests/test_op_normalizer.py`

- [ ] **Step 1: 写失败测试**

`tests/test_op_normalizer.py`:
```python
from opseq.op_normalizer import normalize_op_type


def test_known_matmul_variants():
    assert normalize_op_type("MatMul") == "MatMul"
    assert normalize_op_type("MatMulV2") == "MatMul"
    assert normalize_op_type("BatchMatMulV2") == "MatMul"


def test_known_others():
    assert normalize_op_type("LayerNorm") == "LayerNorm"
    assert normalize_op_type("SoftmaxV2") == "Softmax"
    assert normalize_op_type("EmbeddingBag") == "TBE_Lookup"
    assert normalize_op_type("GatherV2") == "Gather"
    assert normalize_op_type("ReduceMeanD") == "Reduction"
    assert normalize_op_type("Add") == "FusedElementwise"


def test_unknown_falls_back():
    assert normalize_op_type("SomeBrandNewOp") == "Unknown"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_op_normalizer.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.op_normalizer'`

- [ ] **Step 3: 实现 op_normalizer.py**

`src/opseq/op_normalizer.py`:
```python
"""GE 算子类型 → 归一化 op_type。未覆盖的落 Unknown，不阻塞主链路。"""

_GE_TO_NORM: dict[str, str] = {
    # MatMul 族
    "MatMul": "MatMul",
    "MatMulV2": "MatMul",
    "BatchMatMul": "MatMul",
    "BatchMatMulV2": "MatMul",
    # 归一化
    "LayerNorm": "LayerNorm",
    "LayerNormV2": "LayerNorm",
    # Softmax
    "Softmax": "Softmax",
    "SoftmaxV2": "Softmax",
    # embedding 查表（opaque）
    "EmbeddingBag": "TBE_Lookup",
    "Gather": "Gather",
    "GatherV2": "Gather",
    # 拼接
    "Concat": "Concat",
    "ConcatD": "Concat",
    "ConcatV2": "Concat",
    # 规约
    "ReduceSum": "Reduction",
    "ReduceSumD": "Reduction",
    "ReduceMean": "Reduction",
    "ReduceMeanD": "Reduction",
    # 逐元素（融合）
    "Add": "FusedElementwise",
    "Mul": "FusedElementwise",
    "Sub": "FusedElementwise",
    "Relu": "FusedElementwise",
    "Sigmoid": "FusedElementwise",
    "Tanh": "FusedElementwise",
}


def normalize_op_type(ge_op_type: str) -> str:
    return _GE_TO_NORM.get(ge_op_type, "Unknown")
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_op_normalizer.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add src/opseq/op_normalizer.py tests/test_op_normalizer.py
git commit -m "feat: GE 算子类型归一化映射"
```

---

## Task 3: GE 节点解析 → IR

**Files:**
- Create: `src/opseq/ge_parser.py`
- Test: `tests/test_ge_parser.py`

**约定**：`ge_parser` 消费一个「节点 dict 列表」（由 Task 6 的 `ge_reader` 产出），节点 dict 结构：
```python
{
  "name": "MatMulV2_0",
  "type": "MatMulV2",
  "input_desc":  [{"shape": [1024, 256], "dtype": "DT_FLOAT16", "format": "ND"}, ...],
  "output_desc": [{"shape": [1024, 128], "dtype": "DT_FLOAT16", "format": "ND"}],
  "attr": {"transpose_x2": False, "fusion_group": 0},
}
```

- [ ] **Step 1: 写失败测试**

`tests/test_ge_parser.py`:
```python
from opseq.ge_parser import normalize_dtype, parse_ge_graph
from opseq.ir import InputSpec, InputDesc


def test_normalize_dtype():
    assert normalize_dtype("DT_FLOAT16") == "fp16"
    assert normalize_dtype("DT_FLOAT") == "fp32"
    assert normalize_dtype("DT_BF16") == "bf16"
    assert normalize_dtype("DT_INT64") == "int64"
    assert normalize_dtype("DT_WEIRD") == "dt_weird"  # 兜底小写


def _nodes():
    return [
        {"name": "Data_0", "type": "Data", "input_desc": [],
         "output_desc": [{"shape": [1024, 256], "dtype": "DT_FLOAT16", "format": "ND"}], "attr": {}},
        {"name": "MatMulV2_0", "type": "MatMulV2",
         "input_desc": [{"shape": [1024, 256], "dtype": "DT_FLOAT16", "format": "ND"}],
         "output_desc": [{"shape": [1024, 128], "dtype": "DT_FLOAT16", "format": "FRACTAL_NZ"}],
         "attr": {"transpose_x2": False, "fusion_group": 3}},
        {"name": "NetOutput", "type": "NetOutput", "input_desc": [], "output_desc": [], "attr": {}},
    ]


def _spec():
    return InputSpec(inputs=[InputDesc(name="dense", shape=[1024, 256], dtype="fp16")])


def test_parse_skips_structural_nodes():
    cap = parse_ge_graph(_nodes(), model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    # Data / NetOutput 被跳过，只剩 1 个计算算子
    assert len(cap.ops) == 1
    assert cap.ops[0].op_type == "MatMul"
    assert cap.ops[0].backend_op_name == "MatMulV2_0"


def test_parse_shape_dtype_format():
    cap = parse_ge_graph(_nodes(), model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    op = cap.ops[0]
    assert op.inputs[0].shape == [1024, 256]
    assert op.inputs[0].dtype == "fp16"
    assert op.outputs[0].format == "FRACTAL_NZ"
    assert op.fusion_group_id == 3
    assert op.attrs["transpose_x2"] is False


def test_parse_sequential_ids_and_validates():
    cap = parse_ge_graph(_nodes(), model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    assert cap.ops[0].id == 0
    assert cap.backend == "ascend"
    cap.validate()  # 不抛异常


def test_unknown_fusion_group_defaults_to_id():
    nodes = [{"name": "Relu_0", "type": "Relu",
              "input_desc": [{"shape": [8], "dtype": "DT_FLOAT16", "format": "ND"}],
              "output_desc": [{"shape": [8], "dtype": "DT_FLOAT16", "format": "ND"}],
              "attr": {}}]
    cap = parse_ge_graph(nodes, model_id="m", model_name="M",
                         chip="Ascend910B", input_spec=_spec())
    assert cap.ops[0].fusion_group_id == 0  # 无 fusion_group 时回退为 id
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_ge_parser.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.ge_parser'`

- [ ] **Step 3: 实现 ge_parser.py**

`src/opseq/ge_parser.py`:
```python
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
        fusion_group = attr.get("fusion_group", next_id)
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
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_ge_parser.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add src/opseq/ge_parser.py tests/test_ge_parser.py
git commit -m "feat: GE 节点解析为公共 IR"
```

---

## Task 4: IR 缓存

**Files:**
- Create: `src/opseq/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cache.py`:
```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_cache.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.cache'`

- [ ] **Step 3: 实现 cache.py**

`src/opseq/cache.py`:
```python
"""按 (model_id, shape_hash, backend) 缓存 IR JSON。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from opseq.ir import InputSpec, ModelOpSeqCapture


def shape_hash(spec: InputSpec) -> str:
    payload = [
        {"name": i.name, "shape": i.shape, "dtype": i.dtype, "format": i.format}
        for i in spec.inputs
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class IRCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, model_id: str, shape_hash: str, backend: str) -> str:
        return f"{model_id}__{shape_hash}__{backend}.json"

    def put(self, capture: ModelOpSeqCapture, *, shape_hash: str) -> Path:
        path = self.root / self._key(capture.model_id, shape_hash, capture.backend)
        path.write_text(capture.to_json(indent=2), encoding="utf-8")
        return path

    def get(self, *, model_id: str, shape_hash: str, backend: str) -> Optional[ModelOpSeqCapture]:
        path = self.root / self._key(model_id, shape_hash, backend)
        if not path.exists():
            return None
        return ModelOpSeqCapture.from_json(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_cache.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add src/opseq/cache.py tests/test_cache.py
git commit -m "feat: IR JSON 缓存（按 model_id+shape+backend）"
```

---

## Task 5: specialize 输入构造器

**Files:**
- Create: `src/opseq/input_builder.py`
- Test: `tests/test_input_builder.py`

**说明**：依赖 torch + torchrec（CPU 即可测）。若环境未装 torchrec，测试跳过。

- [ ] **Step 1: 写失败测试**

`tests/test_input_builder.py`:
```python
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
    # lengths 长度 = keys 数 * batch
    assert kjt.lengths().numel() == 2 * 4
    # values 总数 = batch * (3 + 5)
    assert kjt.values().numel() == 4 * (3 + 5)
    assert kjt.keys() == ["user", "item"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_input_builder.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.input_builder'`（若无 torchrec 则 skipped — 此时先安装：`pip install torch torchrec`）

- [ ] **Step 3: 实现 input_builder.py**

`src/opseq/input_builder.py`:
```python
"""按 ShapeConfig 造 specialize 推理输入（dense + 稀疏 KeyedJaggedTensor）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class SparseFeatureConfig:
    keys: list[str]
    pooling_factors: dict[str, int]  # 每个特征固定 bag 大小（specialize）


@dataclass
class ShapeConfig:
    batch_size: int
    dense_dim: int
    sparse: SparseFeatureConfig
    dtype: str = "float16"
    num_embeddings: int = 1000  # 造 indices 的取值上界


def build_inputs(cfg: ShapeConfig) -> dict:
    dtype = getattr(torch, cfg.dtype)
    dense = torch.randn(cfg.batch_size, cfg.dense_dim, dtype=dtype)

    lengths: list[int] = []
    values: list[int] = []
    for key in cfg.sparse.keys:
        pf = cfg.sparse.pooling_factors[key]
        for _ in range(cfg.batch_size):
            lengths.append(pf)
            values.extend(
                int(x) for x in torch.randint(0, cfg.num_embeddings, (pf,)).tolist()
            )

    kjt = KeyedJaggedTensor(
        keys=list(cfg.sparse.keys),
        values=torch.tensor(values, dtype=torch.int64),
        lengths=torch.tensor(lengths, dtype=torch.int64),
    )
    return {"dense": dense, "sparse": kjt}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_input_builder.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add src/opseq/input_builder.py tests/test_input_builder.py
git commit -m "feat: specialize 输入构造器（含 KJT）"
```

---

## Task 6: GE dump 读取器

**Files:**
- Create: `src/opseq/ge_reader.py`
- Create: `tests/fixtures/sample_ge_graph.json`
- Test: `tests/test_ge_reader.py`

**说明**：torchair 可把 GE 图 dump 为 JSON。GE 图 JSON 中 `attr` 是 `[{key, value}]` 列表，本读取器把它归一化成 Task 3 约定的节点 dict（`attr` 为 dict）。

- [ ] **Step 1: 创建 fixture**

`tests/fixtures/sample_ge_graph.json`:
```json
{
  "graph": [
    {
      "name": "main_graph",
      "op": [
        {
          "name": "Data_0",
          "type": "Data",
          "input": [],
          "input_desc": [],
          "output_desc": [{"shape": {"dim": [1024, 256]}, "dtype": "DT_FLOAT16", "layout": "ND"}],
          "attr": []
        },
        {
          "name": "MatMulV2_0",
          "type": "MatMulV2",
          "input": ["Data_0:0"],
          "input_desc": [{"shape": {"dim": [1024, 256]}, "dtype": "DT_FLOAT16", "layout": "ND"}],
          "output_desc": [{"shape": {"dim": [1024, 128]}, "dtype": "DT_FLOAT16", "layout": "FRACTAL_NZ"}],
          "attr": [{"key": "fusion_group", "value": {"i": 3}}, {"key": "transpose_x2", "value": {"b": false}}]
        }
      ]
    }
  ]
}
```

- [ ] **Step 2: 写失败测试**

`tests/test_ge_reader.py`:
```python
from pathlib import Path

from opseq.ge_reader import read_ge_dump

FIX = Path(__file__).parent / "fixtures" / "sample_ge_graph.json"


def test_reads_nodes_in_order():
    nodes = read_ge_dump(FIX)
    assert [n["name"] for n in nodes] == ["Data_0", "MatMulV2_0"]


def test_normalizes_shape_and_attr():
    nodes = read_ge_dump(FIX)
    mm = nodes[1]
    assert mm["type"] == "MatMulV2"
    assert mm["output_desc"][0]["shape"] == [1024, 128]
    assert mm["output_desc"][0]["format"] == "FRACTAL_NZ"
    assert mm["attr"]["fusion_group"] == 3
    assert mm["attr"]["transpose_x2"] is False
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_ge_reader.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.ge_reader'`

- [ ] **Step 4: 实现 ge_reader.py**

`src/opseq/ge_reader.py`:
```python
"""读 torchair dump 的 GE 图 JSON → Task 3 约定的节点 dict 列表（薄、格式耦合）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _attr_value(value: dict) -> Any:
    # GE attr value 是带类型标签的 union：{"i": 3} / {"b": false} / {"s": "..."} / {"f": 1.0}
    for tag in ("i", "b", "s", "f"):
        if tag in value:
            return value[tag]
    return None


def _attrs_to_dict(attr_list: list[dict]) -> dict:
    return {a["key"]: _attr_value(a.get("value", {})) for a in attr_list}


def _desc(d: dict) -> dict:
    return {
        "shape": list(d.get("shape", {}).get("dim", [])),
        "dtype": d.get("dtype", ""),
        "format": d.get("layout", "ND"),
    }


def read_ge_dump(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes: list[dict] = []
    for graph in raw.get("graph", []):
        for op in graph.get("op", []):
            nodes.append(
                {
                    "name": op.get("name", ""),
                    "type": op.get("type", ""),
                    "input_desc": [_desc(x) for x in op.get("input_desc", [])],
                    "output_desc": [_desc(x) for x in op.get("output_desc", [])],
                    "attr": _attrs_to_dict(op.get("attr", [])),
                }
            )
    return nodes
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_ge_reader.py -q`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add src/opseq/ge_reader.py tests/test_ge_reader.py tests/fixtures/sample_ge_graph.json
git commit -m "feat: GE dump JSON 读取器"
```

---

## Task 7: Ascend 抽取器编排

**Files:**
- Create: `src/opseq/ascend_extractor.py`
- Test: `tests/test_ascend_extractor.py`

**说明**：`extract` 编排「编译+dump → reader → parser → cache」。编译+dump 依赖 torch_npu/torchair（仅 Ascend），通过依赖注入（`compile_and_dump` 可替换）让单测用 mock。真实设备集成另见步骤末注释。

- [ ] **Step 1: 写失败测试（mock 编译+dump）**

`tests/test_ascend_extractor.py`:
```python
from pathlib import Path

from opseq.ascend_extractor import extract
from opseq.cache import IRCache, shape_hash
from opseq.ge_reader import read_ge_dump
from opseq.ir import InputSpec, InputDesc

FIX = Path(__file__).parent / "fixtures" / "sample_ge_graph.json"


def _spec():
    return InputSpec(inputs=[InputDesc(name="dense", shape=[1024, 256], dtype="fp16")])


def test_extract_wires_reader_parser_cache(tmp_path):
    cache = IRCache(root=tmp_path)

    # mock 编译+dump：不碰硬件，直接返回 fixture dump 路径
    def fake_compile_and_dump(model, inputs, dump_dir):
        return FIX

    cap = extract(
        model=object(), inputs={}, model_id="dlrm_v1", model_name="DLRM",
        chip="Ascend910B", input_spec=_spec(), dump_dir=tmp_path / "dump",
        compile_and_dump=fake_compile_and_dump, reader=read_ge_dump, cache=cache,
    )

    assert cap.backend == "ascend"
    assert [op.op_type for op in cap.ops] == ["MatMul"]  # Data 被跳过
    # 已写入缓存
    got = cache.get(model_id="dlrm_v1", shape_hash=shape_hash(_spec()), backend="ascend")
    assert got == cap


def test_extract_hits_cache_without_recompiling(tmp_path):
    cache = IRCache(root=tmp_path)
    calls = {"n": 0}

    def fake_compile_and_dump(model, inputs, dump_dir):
        calls["n"] += 1
        return FIX

    kwargs = dict(
        model=object(), inputs={}, model_id="dlrm_v1", model_name="DLRM",
        chip="Ascend910B", input_spec=_spec(), dump_dir=tmp_path / "dump",
        compile_and_dump=fake_compile_and_dump, reader=read_ge_dump, cache=cache,
    )
    extract(**kwargs)
    extract(**kwargs)  # 第二次应命中缓存
    assert calls["n"] == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_ascend_extractor.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.ascend_extractor'`

- [ ] **Step 3: 实现 ascend_extractor.py**

`src/opseq/ascend_extractor.py`:
```python
"""Ascend 抽取器编排：编译+dump → reader → parser → cache。

编译+dump 默认实现依赖 torch_npu/torchair（仅 Ascend）。单测通过传入
compile_and_dump 替身绕开硬件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from opseq.cache import IRCache, shape_hash
from opseq.ge_parser import parse_ge_graph
from opseq.ge_reader import read_ge_dump
from opseq.ir import InputSpec, ModelOpSeqCapture


def _default_compile_and_dump(model, inputs: dict, dump_dir: Path) -> Path:
    """在 Ascend 上用 torchair 编译并 dump GE 图，返回 dump 文件路径。

    仅在 NPU 环境可用。优先 FakeTensor/export 模式只构图不真执行 kernel；
    若不可用，退化为触发一次真实 forward（仍是一次性，非 profiling 循环）。
    """
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
    import torchair

    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    config = torchair.CompilerConfig()
    config.debug.graph_dump.type = "json"
    config.debug.graph_dump.path = str(dump_dir)
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    compiled = torch.compile(model, backend=npu_backend, dynamic=False)
    with torch.no_grad():
        compiled(**inputs)  # 触发一次编译+dump

    dumps = sorted(dump_dir.glob("*.json"))
    if not dumps:
        raise FileNotFoundError(f"未在 {dump_dir} 找到 GE dump 文件")
    return dumps[-1]


def extract(
    *,
    model,
    inputs: dict,
    model_id: str,
    model_name: str,
    chip: str,
    input_spec: InputSpec,
    dump_dir: Path,
    compile_and_dump: Callable[[object, dict, Path], Path] = _default_compile_and_dump,
    reader: Callable[[Path], list[dict]] = read_ge_dump,
    cache: Optional[IRCache] = None,
) -> ModelOpSeqCapture:
    h = shape_hash(input_spec)

    if cache is not None:
        cached = cache.get(model_id=model_id, shape_hash=h, backend="ascend")
        if cached is not None:
            return cached

    dump_path = compile_and_dump(model, inputs, Path(dump_dir))
    nodes = reader(dump_path)
    capture = parse_ge_graph(
        nodes, model_id=model_id, model_name=model_name,
        chip=chip, input_spec=input_spec, backend="ascend",
    )

    if cache is not None:
        cache.put(capture, shape_hash=h)
    return capture
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_ascend_extractor.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 设备集成验证（仅 Ascend，手动）**

在 Ascend NPU 机器上跑一次端到端：用 `input_builder.build_inputs` 造输入，调 `extract(...)`（用默认 `compile_and_dump`），确认产出 IR 且 `ops` 非空、shape/dtype/format 合理。记录 `_default_compile_and_dump` 中 torchair 配置项（`graph_dump.type`/`path`）在当前 CANN/torch_npu 版本下的真实字段名，按需修正。

- [ ] **Step 6: 提交**

```bash
git add src/opseq/ascend_extractor.py tests/test_ascend_extractor.py
git commit -m "feat: Ascend 抽取器编排（编译+dump→解析→缓存）"
```

---

## Task 8: profiler 耗时回填（可选工具）

**Files:**
- Create: `src/opseq/profiler_attach.py`
- Create: `tests/fixtures/sample_profiler.csv`
- Test: `tests/test_profiler_attach.py`

**说明**：解析 + join 为纯逻辑、可测；真实跑模型采集 profiler 在 Ascend 上做。`attach_measured` 不改原 IR，返回新副本。

- [ ] **Step 1: 创建 fixture**

`tests/fixtures/sample_profiler.csv`:
```csv
Op Name,Task Duration(us)
MatMulV2_0,12.5
Relu_0,1.2
```

- [ ] **Step 2: 写失败测试**

`tests/test_profiler_attach.py`:
```python
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
    # 未匹配到的保持 None
    assert out.ops[1].measured is None
    # 原 IR 不被修改
    assert cap.ops[0].measured is None
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_profiler_attach.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'opseq.profiler_attach'`

- [ ] **Step 4: 实现 profiler_attach.py**

`src/opseq/profiler_attach.py`:
```python
"""可选工具：解析 Ascend profiler 输出，按 backend_op_name 回填 measured。"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

from opseq.ir import Measured, ModelOpSeqCapture


def parse_profiler(path: str | Path) -> dict[str, float]:
    """解析 msprof 导出的 op summary CSV → {op_name: latency_us}。"""
    latencies: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Op Name", "").strip()
            dur = row.get("Task Duration(us)", "").strip()
            if name and dur:
                latencies[name] = float(dur)
    return latencies


def attach_measured(
    capture: ModelOpSeqCapture, latencies: dict[str, float]
) -> ModelOpSeqCapture:
    """返回回填了 measured 的新 IR 副本，不修改入参。"""
    out = copy.deepcopy(capture)
    for op in out.ops:
        if op.backend_op_name in latencies:
            op.measured = Measured(latency_us=latencies[op.backend_op_name])
    return out
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_profiler_attach.py -q`
Expected: PASS（2 passed）

- [ ] **Step 6: 设备集成验证（仅 Ascend，手动）**

在 Ascend 上用 `torch_npu.profiler` 跑 N 次目标模型，导出 op summary，确认列名与 `parse_profiler` 的 `"Op Name"`/`"Task Duration(us)"` 一致（不同版本列名可能不同，按需调整），并验证 `backend_op_name` 与 profiler 的 kernel 名匹配率。匹配不准时按「名字+执行序+shape」多键匹配（见 spec 风险表）。

- [ ] **Step 7: 提交**

```bash
git add src/opseq/profiler_attach.py tests/test_profiler_attach.py tests/fixtures/sample_profiler.csv
git commit -m "feat: profiler 耗时回填可选工具"
```

---

## Task 9: 全量回归与收尾

- [ ] **Step 1: 跑全部单测**

Run: `python -m pytest -q`
Expected: 全部 PASS（无 NPU 环境下 `test_input_builder` 在装了 torchrec 时 PASS，否则 skipped）

- [ ] **Step 2: 确认无残留 NPU 依赖泄漏到纯逻辑测试**

Run: `python -m pytest -q -k "not input_builder"`
Expected: 在仅装标准库 + pytest 的环境下全部 PASS（验证纯逻辑层零硬件依赖）

- [ ] **Step 3: 提交（如有收尾改动）**

```bash
git add -A
git commit -m "test: 全量回归通过" || echo "无改动"
```
