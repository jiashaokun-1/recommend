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
