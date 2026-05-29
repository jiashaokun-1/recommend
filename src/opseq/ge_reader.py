"""读 torchair dump 的 GE 图 → Task 3 约定的节点 dict 列表（薄、格式耦合）。

torchair 的 ``config.debug.graph_dump.type`` 只支持 ``txt`` / ``pbtxt`` / ``py``
（无 ``json``）。``txt`` 是 GE GraphDef 的 protobuf text 格式，结构与本项目约定的
节点模型一一对应，故这里实现一个轻量 textproto 解析器，不引入 protobuf 依赖。

融合后的 optimized 图里，许多中间张量的 ``shape {}`` 字段会被省略，真实静态 shape
落在 torch fx 注入的 ``_meta`` 属性串里（形如
``Tensor(dtype=torch.float32, shape=torch.Size([4, 16])``）。reader 在 ``shape {}``
缺失时回退到 ``_meta`` 提取 shape/dtype，保证 roofline 需要的形状不丢。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# textproto 词法：花括号、冒号、双引号串（含转义）、裸 token（标识符/数字/枚举）
_TOKEN = re.compile(
    r'\s+|(?P<lb>\{)|(?P<rb>\})|(?P<colon>:)'
    r'|(?P<string>"(?:\\.|[^"\\])*")'
    r'|(?P<token>[^\s{}:]+)'
)

# torch dtype（出现在 _meta 串里）→ GE dtype 标签，复用 ge_parser 的归一化表
_TORCH_TO_GE_DTYPE = {
    "float32": "DT_FLOAT", "float16": "DT_FLOAT16", "bfloat16": "DT_BF16",
    "int64": "DT_INT64", "int32": "DT_INT32", "int8": "DT_INT8",
    "uint8": "DT_UINT8", "bool": "DT_BOOL",
}

_META_SHAPE = re.compile(r"torch\.Size\(\[([0-9,\s]*)\]")
_META_DTYPE = re.compile(r"dtype=torch\.(\w+)")


def _tokenize(text: str) -> list[tuple[str, str]]:
    return [(m.lastgroup, m.group()) for m in _TOKEN.finditer(text) if m.lastgroup]


def _scalar(kind: str, raw: str) -> Any:
    if kind == "string":
        body = raw[1:-1]
        try:
            return body.encode("latin-1", "backslashreplace").decode("unicode_escape")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return body
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw  # 枚举裸 token，如 DT_FLOAT


def _parse_message(tokens: list[tuple[str, str]], i: int) -> tuple[dict, int]:
    """解析一段 textproto message body，返回 (dict, 下一个 token 下标)。

    重复出现的 key 收敛为 list；遇到 ``}`` 或 token 耗尽即结束。
    """
    result: dict[str, Any] = {}

    def add(key: str, value: Any) -> None:
        if key in result:
            if not isinstance(result[key], list):
                result[key] = [result[key]]
            result[key].append(value)
        else:
            result[key] = value

    while i < len(tokens):
        kind, raw = tokens[i]
        if kind == "rb":
            return result, i + 1
        key = raw
        i += 1
        next_kind, _ = tokens[i]
        if next_kind == "lb":  # key { ... }
            value, i = _parse_message(tokens, i + 1)
        elif next_kind == "colon":
            vk, vr = tokens[i + 1]
            if vk == "lb":  # key: { ... }
                value, i = _parse_message(tokens, i + 2)
            else:  # key: scalar
                value, i = _scalar(vk, vr), i + 2
        else:
            raise ValueError(f"GE dump 解析失败：key {key!r} 后出现意外 token {tokens[i]}")
        add(key, value)
    return result, i


def _as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _shape_dtype_from_meta(desc: dict) -> tuple[list[int] | None, str | None]:
    for attr in _as_list(desc.get("attr")):
        if isinstance(attr, dict) and attr.get("key") == "_meta":
            s = (attr.get("value") or {}).get("s", "")
            shape = dtype = None
            m = _META_SHAPE.search(s)
            if m:
                shape = [int(x) for x in m.group(1).split(",") if x.strip()]
            d = _META_DTYPE.search(s)
            if d:
                dtype = _TORCH_TO_GE_DTYPE.get(d.group(1))
            return shape, dtype
    return None, None


def _desc(desc: dict) -> dict:
    shape_msg = desc.get("shape")
    dims = _as_list(shape_msg.get("dim")) if isinstance(shape_msg, dict) else []
    dtype = desc.get("dtype", "")
    if not dims or not dtype:  # 融合图常缺 shape{}/dtype，回退 fx _meta
        meta_shape, meta_dtype = _shape_dtype_from_meta(desc)
        if not dims and meta_shape is not None:
            dims = meta_shape
        if not dtype and meta_dtype is not None:
            dtype = meta_dtype
    return {
        "shape": [int(x) for x in dims],
        "dtype": dtype,
        "format": desc.get("layout", "ND"),
    }


def _attr_value(value: dict) -> Any:
    # GE attr value 是带类型标签的 union：{"i":3}/{"b":false}/{"s":"..."}/{"f":1.0}
    if isinstance(value, dict):
        for tag in ("i", "b", "s", "f"):
            if tag in value:
                return value[tag]
    return None  # list 等内部记账属性，丢弃


def _attrs(op: dict) -> dict:
    out = {}
    for attr in _as_list(op.get("attr")):
        if isinstance(attr, dict) and "key" in attr:
            v = _attr_value(attr.get("value", {}))
            if v is not None:
                out[attr["key"]] = v
    return out


def read_ge_dump(path: str | Path) -> list[dict]:
    """读 torchair ``txt`` GE dump，返回节点 dict 列表（供 parse_ge_graph 消费）。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    graph, _ = _parse_message(_tokenize(text), 0)
    nodes: list[dict] = []
    for op in _as_list(graph.get("op")):
        nodes.append(
            {
                "name": op.get("name", ""),
                "type": op.get("type", ""),
                "input_desc": [_desc(x) for x in _as_list(op.get("input_desc"))],
                "output_desc": [_desc(x) for x in _as_list(op.get("output_desc"))],
                "attr": _attrs(op),
            }
        )
    return nodes
