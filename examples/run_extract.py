#!/usr/bin/env python3
"""配置驱动的 opseq 算子序抽取启动器。

用法：
    # 用内置 tiny_dlrm 默认配置（开箱即用）
    python examples/run_extract.py

    # 指定配置文件
    python examples/run_extract.py --config examples/config.tiny_dlrm.json

    # 命令行覆盖若干字段
    python examples/run_extract.py --config examples/config.tiny_dlrm.json \
        --device npu:0 --output /tmp/my.ir.json --no-cache

配置（JSON）支持两种取模型方式：
    model.builtin = "tiny_dlrm"       内置示例模型（按 params 参数化）
    model.factory = "pkg.mod:func"    你的模型工厂（返回 nn.Module）

Ascend 上运行前请先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，
可用 `ASCEND_RT_VISIBLE_DEVICES` 选卡。
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
import sys

# 允许未 `pip install -e .` 时直接从源码运行
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 内置默认配置：不带 --config 时即用它跑 tiny_dlrm
DEFAULT_CONFIG = {
    "model": {
        "builtin": "tiny_dlrm",
        "params": {"n_tables": 3, "vocab": 1000, "emb_dim": 16, "dense_dim": 13},
    },
    "inputs": {"batch_size": 4},
    "meta": {
        "model_id": "tiny_dlrm",
        "model_name": "TinyDLRM",
        "chip": "Ascend910",
        "backend": "ascend",
    },
    "runtime": {
        "device": "npu",
        "dump_dir": "/tmp/opseq_dump",
        "cache_dir": "/tmp/opseq_cache",
        "output": "tiny_dlrm.ir.json",
    },
}


def _deep_merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = _deep_merge(base[k], v)
        else:
            out[k] = v
    return out


def _load_obj(spec: str):
    """'package.module:callable' → 可调用对象。"""
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"factory 需写成 'module:callable'，收到 {spec!r}")
    return getattr(importlib.import_module(mod), attr)


# --------------------------------------------------------------------------
# 内置示例模型 tiny_dlrm（与真机验证用的模型一致）
# --------------------------------------------------------------------------
def build_tiny_dlrm(params: dict):
    import torch
    from torch import nn

    n_tables = params.get("n_tables", 3)
    vocab = params.get("vocab", 1000)
    emb_dim = params.get("emb_dim", 16)
    dense_dim = params.get("dense_dim", 13)

    class TinyDLRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList(
                [nn.Embedding(vocab, emb_dim) for _ in range(n_tables)])
            self.bottom = nn.Sequential(
                nn.Linear(dense_dim, 64), nn.ReLU(),
                nn.Linear(64, emb_dim), nn.ReLU())
            self.top = nn.Sequential(
                nn.Linear(emb_dim * (n_tables + 1), 64), nn.ReLU(),
                nn.Linear(64, 1))

        def forward(self, dense, sparse):
            d = self.bottom(dense)
            es = [emb(sparse[i]) for i, emb in enumerate(self.embs)]
            return torch.sigmoid(self.top(torch.cat([d] + es, dim=1)))

    return TinyDLRM()


def build_tiny_dlrm_inputs(params: dict, batch_size: int, device: str) -> dict:
    import torch

    n_tables = params.get("n_tables", 3)
    vocab = params.get("vocab", 1000)
    dense_dim = params.get("dense_dim", 13)
    dense = torch.randn(batch_size, dense_dim)
    sparse = torch.randint(0, vocab, (n_tables, batch_size), dtype=torch.int64)
    return {"dense": dense.to(device), "sparse": sparse.to(device)}


def tiny_dlrm_input_spec(params: dict, batch_size: int):
    from opseq.ir import InputDesc, InputSpec

    return InputSpec(inputs=[
        InputDesc(name="dense", shape=[batch_size, params.get("dense_dim", 13)], dtype="fp32"),
        InputDesc(name="sparse", shape=[params.get("n_tables", 3), batch_size], dtype="int64"),
    ])


def _build(cfg: dict, device: str):
    """按 config 造 (model, inputs, input_spec)。"""
    from opseq.ir import InputDesc, InputSpec

    mcfg = cfg["model"]
    if mcfg.get("builtin") == "tiny_dlrm":
        params = mcfg.get("params", {})
        bs = cfg["inputs"]["batch_size"]
        return (
            build_tiny_dlrm(params),
            build_tiny_dlrm_inputs(params, bs, device),
            tiny_dlrm_input_spec(params, bs),
        )
    if "factory" in mcfg:
        model = _load_obj(mcfg["factory"])(**mcfg.get("params", {}))
        in_fac = cfg["inputs"].get("factory")
        if not in_fac:
            raise SystemExit("factory 模式需在 inputs.factory 提供输入构造函数")
        inputs = _load_obj(in_fac)(**cfg["inputs"].get("params", {}))
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        sp = cfg.get("input_spec")
        if not sp:
            raise SystemExit("factory 模式需在 config 提供 input_spec")
        spec = InputSpec(
            inputs=[InputDesc(**d) for d in sp["inputs"]],
            symbol_bindings=sp.get("symbol_bindings", {}),
        )
        return model, inputs, spec
    raise SystemExit("model 需指定 builtin 或 factory")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="配置驱动的 opseq 算子序抽取启动器")
    ap.add_argument("--config", type=Path, help="JSON 配置路径；缺省用内置 tiny_dlrm")
    ap.add_argument("--device", help="覆盖 runtime.device（npu / npu:0 / cpu）")
    ap.add_argument("--dump-dir", help="覆盖 runtime.dump_dir")
    ap.add_argument("--cache-dir", help="覆盖 runtime.cache_dir")
    ap.add_argument("--output", help="覆盖 runtime.output（IR JSON 落盘路径）")
    ap.add_argument("--model-id", help="覆盖 meta.model_id（影响缓存键）")
    ap.add_argument("--no-cache", action="store_true", help="禁用 IR 缓存")
    ap.add_argument("--limit", type=int, default=0, help="只打印前 N 个算子（0=全部）")
    args = ap.parse_args(argv)

    cfg = _deep_merge(DEFAULT_CONFIG, {})
    if args.config:
        cfg = _deep_merge(cfg, json.loads(Path(args.config).read_text(encoding="utf-8")))
    rt = cfg["runtime"]
    if args.device:
        rt["device"] = args.device
    if args.dump_dir:
        rt["dump_dir"] = args.dump_dir
    if args.cache_dir:
        rt["cache_dir"] = args.cache_dir
    if args.output:
        rt["output"] = args.output
    if args.model_id:
        cfg["meta"]["model_id"] = args.model_id

    device = rt["device"]
    import torch  # noqa: F401
    if device.startswith("npu"):
        import torch_npu  # noqa: F401

    from opseq.ascend_extractor import extract
    from opseq.cache import IRCache

    model, inputs, spec = _build(cfg, device)
    model = model.to(device).eval()

    cache = None if args.no_cache else IRCache(root=rt["cache_dir"])
    capture = extract(
        model=model, inputs=inputs,
        model_id=cfg["meta"]["model_id"], model_name=cfg["meta"]["model_name"],
        chip=cfg["meta"]["chip"], input_spec=spec,
        dump_dir=rt["dump_dir"], cache=cache,
    )

    dist = Counter(op.op_type for op in capture.ops)
    print(f"[opseq] model={cfg['meta']['model_id']} device={device} backend={capture.backend} "
          f"ops={len(capture.ops)}")
    print(f"[opseq] op_type dist: {dict(dist)}")
    rows = capture.ops[: args.limit] if args.limit else capture.ops
    print(f"{'id':>3} {'op_type':<16} {'in0':<16} -> out0")
    for op in rows:
        i0 = f"{op.inputs[0].shape}:{op.inputs[0].dtype}" if op.inputs else "-"
        o0 = (f"{op.outputs[0].shape}:{op.outputs[0].dtype}[{op.outputs[0].format}]"
              if op.outputs else "-")
        print(f"{op.id:>3} {op.op_type:<16} {i0:<16} -> {o0}")

    out = Path(rt["output"])
    out.write_text(capture.to_json(indent=2), encoding="utf-8")
    print(f"[opseq] IR -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
