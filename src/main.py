"""PIPO launcher: `python -m src.main server|client [options]`.

Replaces the old example/ scripts: the picked model and all launch parameters
come from config.toml / CLI arguments.
"""

import argparse

from src.common.config import apply_args, he_engine_args, load_config, set_config
from src.heutil import create_he


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipo", description="PIPO privacy-preserving inference")
    parser.add_argument("role", nargs="?", default="server", choices=["server", "client"],
                        help="run as server or client")
    parser.add_argument("--config", default=None, help="config file (default: config.toml)")
    parser.add_argument("--mode", dest="run_mode", default=argparse.SUPPRESS,
                        choices=["offline", "standard", "online"],
                        help="run mode: offline (generate and persist resources), "
                             "standard (offline + online, resources not persisted), "
                             "online (restore persisted resources)")
    for flag, dest in [
        ("--host", "host"), ("--port", "port"), ("--device", "device"),
        ("--model", "model"), ("--seed", "seed"),
        ("--wfile", "wfile"), ("--wfolder", "wfolder"),
        ("--protocol", "protocol"), ("--pad-ratio", "pad_ratio"),
        ("--res-dir", "res_dir"), ("--res-index", "res_index"),
    ]:
        converter = {"port": int, "seed": int, "pad_ratio": float,
                     "res_index": int}.get(dest)
        if converter:
            parser.add_argument(flag, default=argparse.SUPPRESS, type=converter)
        else:
            parser.add_argument(flag, default=argparse.SUPPRESS)
    parser.add_argument("--n", default=argparse.SUPPRESS, type=int,
                        help="standard/online: number of online rounds; "
                             "offline: number of resource sets to generate")
    parser.add_argument("--use-he", dest="use_he", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--show-detail", dest="show_detail", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--verify", dest="verify", action="store_true", default=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mapping = {
        "common.host": "host",
        "common.port": "port",
        "common.device": "device",
        "common.n": "n",
        "common.mode": "run_mode",
        "common.use_he": "use_he",
        "common.show_detail": "show_detail",
        "common.model.name": "model",
        "common.model.seed": "seed",
        "common.model.wfile": "wfile",
        "common.model.wfolder": "wfolder",
        "common.resource.dir": "res_dir",
        "common.resource.index": "res_index",
        "server.protocol.name": "protocol",
        "server.protocol.pad_ratio": "pad_ratio",
        "client.verify": "verify",
    }
    cfg = apply_args(cfg, args, mapping)
    common = cfg.common
    mode = common.mode
    if mode not in ("offline", "standard", "online"):
        raise ValueError(
            "invalid run mode {!r} (expected offline | standard | online)".format(mode))
    # 'offline' persists the generated resources, 'standard' generates them but
    # does not persist, 'online' only loads them; both roles take the same pair
    # of flags so they run the same subset of phases
    do_offline = mode in ("offline", "standard")
    do_online = mode in ("standard", "online")
    set_config(cfg)

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