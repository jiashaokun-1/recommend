from pathlib import Path

from opseq.ge_reader import read_ge_dump

# 真实 torchair txt（protobuf-text）GE dump：tiny matmul+relu 的 optimized 融合图
FIX = Path(__file__).parent / "fixtures" / "sample_ge_graph.txt"


def test_reads_nodes_in_order():
    nodes = read_ge_dump(FIX)
    assert [n["name"] for n in nodes] == ["arg0_1", "arg1_1", "MatMul", "Relu", "NetOutput"]
    assert [n["type"] for n in nodes] == ["Data", "Data", "MatMul", "Relu", "NetOutput"]


def test_reads_shape_dtype_format_from_shape_field():
    nodes = read_ge_dump(FIX)
    mm = next(n for n in nodes if n["name"] == "MatMul")
    assert mm["type"] == "MatMul"
    assert mm["input_desc"][0]["shape"] == [4, 8]
    assert mm["input_desc"][1]["shape"] == [8, 16]
    assert mm["input_desc"][0]["dtype"] == "DT_FLOAT"
    assert mm["input_desc"][0]["format"] == "ND"


def test_falls_back_to_meta_when_shape_field_absent():
    # 融合图里 MatMul 输出无 shape{}，真实 shape 落在 fx _meta 串
    nodes = read_ge_dump(FIX)
    mm = next(n for n in nodes if n["name"] == "MatMul")
    assert mm["output_desc"][0]["shape"] == [4, 16]
    assert mm["output_desc"][0]["dtype"] == "DT_FLOAT"
    # Relu 的输入/输出也仅靠 _meta
    relu = next(n for n in nodes if n["name"] == "Relu")
    assert relu["input_desc"][0]["shape"] == [4, 16]
    assert relu["output_desc"][0]["shape"] == [4, 16]


def test_unfed_optional_input_has_empty_shape():
    nodes = read_ge_dump(FIX)
    mm = next(n for n in nodes if n["name"] == "MatMul")
    # bias 是未喂入的可选输入：既无 shape{} 也无 _meta
    assert mm["input_desc"][2]["shape"] == []


def test_attr_union_scalar_values_kept_bookkeeping_dropped():
    nodes = read_ge_dump(FIX)
    mm = next(n for n in nodes if n["name"] == "MatMul")
    assert mm["attr"]["transpose_x2"] is False
    assert mm["attr"]["transpose_x1"] is False
    # list 类内部记账属性（_input_name_key 等）应被丢弃，不污染 attr
    assert "_input_name_key" not in mm["attr"]
