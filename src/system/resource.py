"""Resource persistence (run modes 'offline' / 'online').

A resource set is one numbered pair of files under ``<dir>/<model_name>/``:

    res_XXXX_server.pt  res_XXXX_client.pt

Each file is a torch.save payload holding the run identity (version,
model_name, model_info_hash, protocol, protocol_args, role) plus the
per-layer resource entries collected from the layers' protocol objects
(``layer.save_res()``). The server and the client each save and load only
their own file; the pair shares the index, and both identity fields and
entries are validated against the current run before use.
"""

import random
from pathlib import Path

import torch

from src.common.config import get_config
from src.common.model_info_io import model_info_hash
from src.common.types import ModelInfo

VERSION = 1


def _stem(index: int) -> str:
    return "res_{:04d}".format(index)


def model_resource_dir(res_dir, model_name: str) -> Path:
    return Path(res_dir) / model_name


def list_indices(res_dir, model_name: str) -> list:
    """Indices of the complete (server + client) resource pairs on disk."""
    d = model_resource_dir(res_dir, model_name)
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("res_*_server.pt"):
        name = f.name[:-len("_server.pt")]
        if (d / (name + "_client.pt")).exists():
            try:
                out.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(out)


def next_index(res_dir, model_name: str, index: int = None) -> int:
    """Index for a new resource set (mode=offline): explicit (must be free)
    or the next free one after the highest existing index."""
    existing = list_indices(res_dir, model_name)
    if index is not None:
        if index in existing:
            raise FileExistsError(
                "resource set {} already exists for model {!r} under {}".format(
                    index, model_name, res_dir))
        return index
    return existing[-1] + 1 if existing else 1


def pick_index(res_dir, model_name: str, index: int = None) -> int:
    """Index to load (mode=online): explicit or a random available one."""
    available = list_indices(res_dir, model_name)
    if not available:
        raise FileNotFoundError(
            "no resource sets for model {!r} under {}".format(model_name, res_dir))
    if index is not None:
        if index not in available:
            raise FileNotFoundError(
                "resource set {} not available for model {!r} (have: {})".format(
                    index, model_name, available))
        return index
    chosen = random.choice(available)
    print("Resource set picked at random: {} (available: {})".format(chosen, available))
    return chosen


def _identity(model_name: str, info: ModelInfo) -> dict:
    cfg = get_config()
    return {
        "version": VERSION,
        "model_name": model_name,
        "model_info_hash": model_info_hash(info),
        "protocol": cfg.server.protocol.name,
        "protocol_args": cfg.server.protocol.extra_args(),
    }


def _save(res_dir, model_name: str, index: int, info: ModelInfo, role: str,
          entries: list) -> None:
    identity = _identity(model_name, info)
    d = model_resource_dir(res_dir, model_name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / (_stem(index) + "_" + role + ".pt")
    torch.save({**identity, "role": role, "layers": entries}, path)
    print("Resources saved: {} side of set {} under {}".format(role, index, d))


def _load(res_dir, model_name: str, index: int, info: ModelInfo, role: str) -> list:
    path = model_resource_dir(res_dir, model_name) / (_stem(index) + "_" + role + ".pt")
    if not path.exists():
        raise FileNotFoundError("resource file missing: {}".format(path))
    payload = torch.load(path, weights_only=True, map_location="cpu")
    for key, want in _identity(model_name, info).items():
        got = payload.get(key)
        if got != want:
            raise ValueError(
                "resource field {!r} mismatch in {}: expected {!r}, got {!r}".format(
                    key, path.name, want, got))
    if payload.get("role") != role:
        raise ValueError(
            "resource role mismatch in {}: expected {!r}, got {!r}".format(
                path.name, role, payload.get("role")))
    return payload["layers"]


def save_resource_set_server(res_dir, model_name: str, index: int, info: ModelInfo,
                             entries: list) -> None:
    """Write the server side of one resource set."""
    _save(res_dir, model_name, index, info, "server", entries)


def save_resource_set_client(res_dir, model_name: str, index: int, info: ModelInfo,
                             entries: list) -> None:
    """Write the client side of one resource set."""
    _save(res_dir, model_name, index, info, "client", entries)


def load_resource_set_server(res_dir, model_name: str, index: int, info: ModelInfo) -> list:
    """Load the server side of one resource set, validating the identity
    against the current run (model structure, protocol variant/parameters)."""
    return _load(res_dir, model_name, index, info, "server")


def load_resource_set_client(res_dir, model_name: str, index: int, info: ModelInfo) -> list:
    """Load the client side of one resource set, validating the identity
    against the current run (model structure, protocol variant/parameters)."""
    return _load(res_dir, model_name, index, info, "client")