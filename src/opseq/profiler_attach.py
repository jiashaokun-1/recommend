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
