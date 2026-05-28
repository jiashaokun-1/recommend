"""按 (model_id, shape_hash, backend) 缓存 IR JSON。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from opseq.ir import InputSpec, ModelOpSeqCapture


def shape_hash(spec: InputSpec) -> str:
    payload = [
        {"name": i.name, "shape": i.shape, "dtype": i.dtype, "format": i.format}
        for i in spec.inputs
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class IRCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, model_id: str, shape_hash: str, backend: str) -> str:
        return f"{model_id}__{shape_hash}__{backend}.json"

    def put(self, capture: ModelOpSeqCapture, *, shape_hash: str) -> Path:
        path = self.root / self._key(capture.model_id, shape_hash, capture.backend)
        path.write_text(capture.to_json(indent=2), encoding="utf-8")
        return path

    def get(self, *, model_id: str, shape_hash: str, backend: str) -> Optional[ModelOpSeqCapture]:
        path = self.root / self._key(model_id, shape_hash, backend)
        if not path.exists():
            return None
        return ModelOpSeqCapture.from_json(path.read_text(encoding="utf-8"))
