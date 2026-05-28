"""按 ShapeConfig 造 specialize 推理输入（dense + 稀疏 KeyedJaggedTensor）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class SparseFeatureConfig:
    keys: list[str]
    pooling_factors: dict[str, int]  # 每个特征固定 bag 大小（specialize）


@dataclass
class ShapeConfig:
    batch_size: int
    dense_dim: int
    sparse: SparseFeatureConfig
    dtype: str = "float16"
    num_embeddings: int = 1000  # 造 indices 的取值上界


def build_inputs(cfg: ShapeConfig) -> dict:
    dtype = getattr(torch, cfg.dtype)
    dense = torch.randn(cfg.batch_size, cfg.dense_dim, dtype=dtype)

    lengths: list[int] = []
    values: list[int] = []
    for key in cfg.sparse.keys:
        pf = cfg.sparse.pooling_factors[key]
        for _ in range(cfg.batch_size):
            lengths.append(pf)
            values.extend(
                int(x) for x in torch.randint(0, cfg.num_embeddings, (pf,)).tolist()
            )

    kjt = KeyedJaggedTensor(
        keys=list(cfg.sparse.keys),
        values=torch.tensor(values, dtype=torch.int64),
        lengths=torch.tensor(lengths, dtype=torch.int64),
    )
    return {"dense": dense, "sparse": kjt}
