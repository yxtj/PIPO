"""PIPO launcher: `python -m src.main server|client [options]`."""

import argparse

from src.common.config import add_config_args, apply_args, he_engine_args, load_config
from src.heutil import create_he


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipo", description="PIPO privacy-preserving inference")
    parser.add_argument("role", nargs="?", choices=["server", "client"],
                        help="run as server or client")
    parser.add_argument("--config", default=None, help="config file (default: config.toml)")
    add_config_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_args(cfg, args)
    common = cfg.common
    # offline: generate and persist resources; standard: offline + online,
    # not persisted; online: only restore
    if common.mode not in ("offline", "standard", "online"):
        raise ValueError("invalid run mode {!r} (offline | standard | online)".format(common.mode))
    do_offline = common.mode in ("offline", "standard")
    do_online = common.mode in ("standard", "online")

    if args.role == "server":
        # late imports so that the protocol variant is chosen from the final config
        from models import registry
        from src.system.runner import run_server

        model_cfg = common.model
        inshape, model = registry.load(
            model_cfg.name,
            seed=model_cfg.seed,
            wfile=model_cfg.wfile or None,
            wfolder=model_cfg.wfolder or None,
            device=common.device,
        )
        print("Model loaded: {} on {}".format(model_cfg.name, common.device))
        run_server(common.host, common.port, model, inshape, common.n, cfg,
                   do_offline=do_offline, do_online=do_online, model_name=model_cfg.name)
    else:
        from src.system.runner import run_client

        backend, params = he_engine_args(cfg)
        he = create_he(backend, params=params, init_keys=True)
        run_client(common.host, common.port, he, cfg,
                   do_offline=do_offline, do_online=do_online,
                   dataset=None, n=common.n, verify=cfg.client.verify)


if __name__ == "__main__":
    main()
