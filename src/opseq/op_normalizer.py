"""GE 算子类型 → 归一化 op_type。未覆盖的落 Unknown，不阻塞主链路。"""

_GE_TO_NORM: dict[str, str] = {
    # MatMul 族
    "MatMul": "MatMul",
    "MatMulV2": "MatMul",
    "BatchMatMul": "MatMul",
    "BatchMatMulV2": "MatMul",
    # 归一化
    "LayerNorm": "LayerNorm",
    "LayerNormV2": "LayerNorm",
    # Softmax
    "Softmax": "Softmax",
    "SoftmaxV2": "Softmax",
    # embedding 查表（opaque）
    "EmbeddingBag": "TBE_Lookup",
    "Gather": "Gather",
    "GatherV2": "Gather",
    # 拼接
    "Concat": "Concat",
    "ConcatD": "Concat",
    "ConcatV2": "Concat",
    # 规约
    "ReduceSum": "Reduction",
    "ReduceSumD": "Reduction",
    "ReduceMean": "Reduction",
    "ReduceMeanD": "Reduction",
    # 逐元素（融合）
    "Add": "FusedElementwise",
    "Mul": "FusedElementwise",
    "Sub": "FusedElementwise",
    "Relu": "FusedElementwise",
    "Sigmoid": "FusedElementwise",
    "Tanh": "FusedElementwise",
}


def normalize_op_type(ge_op_type: str) -> str:
    return _GE_TO_NORM.get(ge_op_type, "Unknown")
