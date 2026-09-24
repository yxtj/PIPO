"""ModelInfo JSON encoding, hashing, and file persistence.

Model info crosses three boundaries, all using one canonical JSON encoding so a
hash computed anywhere is comparable anywhere:

- the wire (`MODEL_INFO_RES`, JSON payload),
- the server-side cache file (loaded at startup to derive the handshake hash),
- the client-side cache file (avoids re-requesting model info across runs).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.common.types import ExecutionStep, ModelInfo


def _step_to_dict(step: ExecutionStep) -> dict:
    return {
        "sources": list(step.sources),
        "dest": step.dest,
        "op": step.op,
        "layer_idx": step.layer_idx,
        "k": list(step.k) if step.k else None,
        "s": list(step.s) if step.s else None,
        "p": list(step.p) if step.p else None,
        "concat_dim": step.concat_dim,
    }


def _step_from_dict(d: dict) -> ExecutionStep:
    return ExecutionStep(
        sources=[int(s) for s in d["sources"]],
        dest=int(d["dest"]),
        op=d["op"],
        layer_idx=int(d["layer_idx"]) if d.get("layer_idx") is not None else None,
        k=tuple(d["k"]) if d.get("k") else None,
        s=tuple(d["s"]) if d.get("s") else None,
        p=tuple(d["p"]) if d.get("p") else None,
        concat_dim=int(d["concat_dim"]) if d.get("concat_dim") is not None else None,
    )


def model_info_to_dict(info: ModelInfo) -> dict:
    """Canonical JSON-able dict for a ModelInfo (stable key order for hashing)."""
    return {
        "steps": [_step_to_dict(s) for s in info.steps],
        "shapes": [list(sh) for sh in info.shapes],
        "inshape": list(info.inshape),
    }


def model_info_from_dict(d: dict) -> ModelInfo:
    return ModelInfo(
        steps=[_step_from_dict(s) for s in d["steps"]],
        shapes=[tuple(sh) for sh in d["shapes"]],
        inshape=tuple(d["inshape"]),
    )


def model_info_to_json(info: ModelInfo) -> str:
    return json.dumps(model_info_to_dict(info), sort_keys=True, separators=(",", ":"))


def model_info_from_json(text: str) -> ModelInfo:
    return model_info_from_dict(json.loads(text))


def model_info_hash(info: ModelInfo) -> str:
    """sha256 over the canonical JSON encoding (deterministic across processes)."""
    return hashlib.sha256(model_info_to_json(info).encode("utf-8")).hexdigest()


def model_info_cache_path(model_info_dir: str | Path, model_name: str) -> Path:
    """Path of the per-model info file inside a cache directory: ``<dir>/<model>.json``."""
    return Path(model_info_dir) / f"{model_name}.json"


def save_model_info_file(info: ModelInfo, model_name: str, path: str | Path) -> str:
    """Write ``{model_name, info}`` and return the info hash (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"model_name": model_name, "info": model_info_to_dict(info)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, sort_keys=True, indent=2)
    return model_info_hash(info)


def load_model_info_file(path: str | Path) -> tuple[str, ModelInfo, str] | None:
    """Load ``(model_name, info, hash)`` from a model-info JSON file.

    Returns ``None`` when the file is missing or not a valid model-info record
    (the hash is recomputed from the stored info, not trusted from the file).
    """
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        model_name = record["model_name"]
        info = model_info_from_dict(record["info"])
    except (KeyError, TypeError, ValueError):
        return None
    return model_name, info, model_info_hash(info)
