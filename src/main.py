"""PIPO launcher: `python -m src.main <role> [mode] [options]`."""

import argparse

from src.common.config import add_config_args, apply_args, he_engine_args, load_config
from src.heutil import create_he
from src.system.runner import run_server, run_client

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipo", description="PIPO privacy-preserving inference")
    parser.add_argument("role", choices=["server", "client"],
                        help="run as server or client")
    parser.add_argument("mode", nargs="?", default=argparse.SUPPRESS,
                        choices=["offline", "standard", "online"],
                        help="offline | standard | online (default: from config, usually standard)")
    parser.add_argument("--config", default=None, help="config file (default: config.toml)")
    add_config_args(parser)
    return parser.parse_args()


def print_startup(role: str, cfg):
    """One-line startup summary."""
    common = cfg.common
    he = common.he.backend if common.use_he else "off"
    line = "PIPO {} | mode={} | model={} | protocol={} | device={} | he={} | n={}".format(
        role, common.mode, common.model.name, cfg.server.protocol.name,
        common.device, he, common.n)
    if role == "client" and cfg.client.verify:
        line += " | verify=on"
    print(line)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_args(cfg, args)
    # positional mode: overrides the config value only when given on the CLI
    if hasattr(args, "mode"):
        cfg.common.mode = args.mode
    common = cfg.common
    # offline: generate and persist resources; standard: offline + online,
    # not persisted; online: only restore
    if common.mode not in ("offline", "standard", "online"):
        raise ValueError("invalid run mode {!r} (offline | standard | online)".format(common.mode))
    do_offline = common.mode in ("offline", "standard")
    do_online = common.mode in ("standard", "online")

    print_startup(args.role, cfg)

    if args.role == "server":
        run_server(common.host, common.port, common.n, cfg,
                   do_offline=do_offline, do_online=do_online)
    else:
        backend, params = he_engine_args(cfg)
        he = create_he(backend, params=params, init_keys=True)
        run_client(common.host, common.port, he, cfg,
                   do_offline=do_offline, do_online=do_online,
                   dataset=None, n=common.n, verify=cfg.client.verify)


if __name__ == "__main__":
    main()
