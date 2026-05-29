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


def test_layout_and_reshape_ops():
    # 真机 DLRM 融合图里 Linear 权重前置 Transpose、interaction Reshape 均出现过
    assert normalize_op_type("Transpose") == "Transpose"
    assert normalize_op_type("TransposeD") == "Transpose"
    assert normalize_op_type("Reshape") == "Reshape"
    assert normalize_op_type("TransData") == "TransData"
    assert normalize_op_type("Cast") == "Cast"


def test_unknown_falls_back():
    assert normalize_op_type("SomeBrandNewOp") == "Unknown"
