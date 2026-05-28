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
