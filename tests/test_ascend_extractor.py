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

    def fake_compile_and_dump(model, inputs, dump_dir):
        return FIX

    cap = extract(
        model=object(), inputs={}, model_id="dlrm_v1", model_name="DLRM",
        chip="Ascend910B", input_spec=_spec(), dump_dir=tmp_path / "dump",
        compile_and_dump=fake_compile_and_dump, reader=read_ge_dump, cache=cache,
    )

    assert cap.backend == "ascend"
    assert [op.op_type for op in cap.ops] == ["MatMul"]
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
    extract(**kwargs)
    assert calls["n"] == 1
