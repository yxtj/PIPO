"""TOML config + CLI args for the PIPO client/server CLIs.

``config.toml`` at the project root drives both roles (custom path via
``--config``); CLI flags override it. The ``Config`` is a plain value
object: load it once and pass it around explicitly (no global singleton).

CLI flags are declared once in ``CONFIG_ARGS``: dest derives from the flag
(``-`` -> ``_``), the type from the dataclass default at the config path
(bool -> store_true).
"""

import argparse
import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.toml"


@dataclass
class ModelConfig:
    name: str = "poc-1"
    seed: int = 0
    # weights file; empty = search wfolder, then registry default
    wfile: str = ""
    # default dir for <model_name>.pt / <model_name>.pth
    wfolder: str = "pretrained"


@dataclass
class ProtocolConfig:
    # plaintext | scale | shuffle | pad | padshuffle | scaleshuffle |
    # padscaleshuffle | noise
    name: str = "scale"
    # pad ratio for the pad protocols (0 = no pads, 1 = as many pads as real)
    pad_ratio: float = 0.0
    # Gaussian noise sigma for the noise protocol
    noise_sigma: float = 0.0

    def extra_args(self) -> dict:
        """Protocol kwargs persisted into offline resource files."""
        if self.name in ("pad", "padshuffle", "padscaleshuffle"):
            return {"pad_ratio": self.pad_ratio}
        if self.name == "noise":
            return {"noise_sigma": self.noise_sigma}
        return {}


@dataclass
class ResourceConfig:
    dir: str = "resources"
    # 0 = random set when loading / next free when generating
    index: int = 0


@dataclass
class HEBackendPyfhelConfig:
    # CKKS context parameters ('pyfhel' backend only)
    n: int = 8192
    scale: int = 2 ** 30
    qi_sizes: list = field(default_factory=lambda: [30] * 5)


@dataclass
class HEConfig:
    # backend when use_he: "none" | "pyfhel" (params in [common.he.pyfhel])
    backend: str = "pyfhel"
    pyfhel: HEBackendPyfhelConfig = field(default_factory=HEBackendPyfhelConfig)


@dataclass
class CommonConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    device: str = "cpu"
    # standard/online: online rounds; offline: resource sets per run
    n: int = 1
    # offline | standard | online
    mode: str = "standard"
    # HE in the offline phase (backend via common.he)
    use_he: bool = False
    # print the per-layer statistics before the summary table
    show_detail: bool = False
    he: HEConfig = field(default_factory=HEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)


@dataclass
class ClientConfig:
    verify: bool = False


@dataclass
class ServerConfig:
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)


@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _set_path(obj, path: str, value) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _config_type(path: str):
    obj = Config()
    for part in path.split("."):
        obj = getattr(obj, part)
    return type(obj)


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
    for section in ("common", "client", "server"):
        _merge(data.get(section, {}), getattr(cfg, section))
    return cfg


# CLI flags overriding config values: (flag, dotted config path[, argparse extras])
CONFIG_ARGS = [
    ("--host", "common.host"),
    ("--port", "common.port"),
    ("--device", "common.device"),
    ("--n", "common.n"),
    ("--model", "common.model.name"),
    ("--seed", "common.model.seed"),
    ("--wfile", "common.model.wfile"),
    ("--wfolder", "common.model.wfolder"),
    ("--res-dir", "common.resource.dir"),
    ("--res-index", "common.resource.index"),
    ("--protocol", "server.protocol.name"),
    ("--pad-ratio", "server.protocol.pad_ratio"),
    ("--show-detail", "common.show_detail"),
    ("--use-he", "common.use_he"),
    ("--verify", "client.verify"),
]


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Register the CONFIG_ARGS flags (SUPPRESS default: only typed flags land in args)."""
    for flag, path, *extra in CONFIG_ARGS:
        kwargs = dict(extra[0]) if extra else {}
        kwargs["default"] = argparse.SUPPRESS
        typ = _config_type(path)
        if typ is bool:
            parser.add_argument(flag, dest=flag[2:].replace("-", "_"), action="store_true", **kwargs)
        else:
            parser.add_argument(flag, dest=flag[2:].replace("-", "_"), type=typ, **kwargs)


def apply_args(config: Config, args: argparse.Namespace) -> Config:
    """Copy CLI-provided values over the merged config."""
    for flag, path, *_ in CONFIG_ARGS:
        dest = flag[2:].replace("-", "_")
        if hasattr(args, dest):
            _set_path(config, path, getattr(args, dest))
    return config


def he_engine_args(cfg: Config) -> tuple:
    """(backend, params) for heutil.create_he; use_he=False -> 'none'.
    Kept here so the engine package stays config-free."""
    if not cfg.common.use_he:
        return "none", None
    backend = cfg.common.he.backend
    p = cfg.common.he.pyfhel
    params = {"n": p.n, "scale": p.scale, "qi_sizes": list(p.qi_sizes)} if backend == "pyfhel" else None
    return backend, params
