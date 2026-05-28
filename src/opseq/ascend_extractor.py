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
        compiled(**inputs)

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
