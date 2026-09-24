"""TOML configuration for the PIPO client/server CLIs.

``config.toml`` at the project root is the default config file; a custom path
can be passed via ``--config``. One file drives both roles:

- ``[common]`` - shared settings (host, port, device, model, protocol, HE, ...).
- ``[client]`` - client settings.
- ``[server]`` - server settings.

Command-line arguments take precedence over the file: config-backed args are
built with ``default=argparse.SUPPRESS`` so only flags the user actually typed
land in the namespace, and ``apply_args`` copies them over the merged config
(dotted paths like ``common.model``).
"""

import argparse
import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.toml"


@dataclass
class CommonConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    device: str = "cpu"
    model: str = "poc-1"
    n: int = 1
    seed: int = 0
    # whether to use HE (Pyfhel) in the offline phase
    use_he: bool = False
    # protocol variant: plaintext, scale, shuffle, noise
    protocol: str = "scale"
    # weights file (state dict); empty = registry default / random weights
    wfile: str = ""


@dataclass
class ClientConfig:
    verify: bool = False


@dataclass
class ServerConfig:
    pass


@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _merge(data: dict, target) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            sub = getattr(target, key, None)
            if dataclasses.is_dataclass(sub) and not isinstance(sub, type):
                _merge(value, sub)
        elif hasattr(target, key):
            setattr(target, key, value)


def load_config(path=None) -> Config:
    cfg = Config()
    if path is None:
        if not DEFAULT_CONFIG_FILE.exists():
            return cfg
        path = DEFAULT_CONFIG_FILE
    with open(path, "rb") as f:
        data = tomllib.load(f)
    _merge(data.get("common", {}), cfg.common)
    _merge(data.get("client", {}), cfg.client)
    _merge(data.get("server", {}), cfg.server)
    return cfg


def _set_path(obj, path: str, value) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def apply_args(config: Config, args: argparse.Namespace, mapping: dict[str, str]) -> Config:
    """Copy CLI-provided values (present in ``args``) over the merged config."""
    for path, dest in mapping.items():
        if not hasattr(args, dest):
            continue
        _set_path(config, path, getattr(args, dest))
    return config


# The effective config is set by the entry point (after CLI overrides are
# applied) and read by library modules (protocol dispatch, layer base) that
# need USE_HE / PROTOCOL without going through main().
_config: Config | None = None


def set_config(cfg: Config) -> None:
    global _config
    _config = cfg


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
