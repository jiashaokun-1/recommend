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
