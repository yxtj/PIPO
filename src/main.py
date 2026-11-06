"""PIPO launcher: `python -m src.main server|client [options]`.

Replaces the old example/ scripts: the picked model and all launch parameters
come from config.toml / CLI arguments.
"""

import argparse

from src.common.config import apply_args, he_engine_args, load_config, set_config
from src.heutil import create_he


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipo", description="PIPO privacy-preserving inference")
    parser.add_argument("mode", choices=["server", "client"], help="run as server or client")
    parser.add_argument("--config", default=None, help="config file (default: config.toml)")
    for flag, dest in [
        ("--host", "host"), ("--port", "port"), ("--device", "device"),
        ("--model", "model"), ("--n", "n"), ("--seed", "seed"),
        ("--wfile", "wfile"), ("--wfolder", "wfolder"),
        ("--protocol", "protocol"), ("--pad-ratio", "pad_ratio"),
    ]:
        converter = {"port": int, "n": int, "seed": int, "pad_ratio": float}.get(dest)
        if converter:
            parser.add_argument(flag, default=argparse.SUPPRESS, type=converter)
        else:
            parser.add_argument(flag, default=argparse.SUPPRESS)
    parser.add_argument("--use-he", dest="use_he", action="store_true", default=argparse.SUPPRESS)
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
        "common.use_he": "use_he",
        "common.model.name": "model",
        "common.model.seed": "seed",
        "common.model.wfile": "wfile",
        "common.model.wfolder": "wfolder",
        "server.protocol.name": "protocol",
        "server.protocol.pad_ratio": "pad_ratio",
        "client.verify": "verify",
    }
    cfg = apply_args(cfg, args, mapping)
    common = cfg.common
    set_config(cfg)

    # the client owns the keyed engine; the server builds its shared engine
    # in Server.setup(). Both resolve the backend/params from the config via
    # he_engine_args; missing Pyfhel raises a clear ImportError.
    he = None
    if args.mode == "client":
        backend, params = he_engine_args(cfg)
        he = create_he(backend, params=params, init_keys=True)

    if args.mode == "server":
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
        run_server(common.host, common.port, model, inshape, common.n,
                   model_name=model_cfg.name)
    else:
        from src.system.runner import run_client

        run_client(common.host, common.port, he,
                   dataset=None, n=common.n, verify=cfg.client.verify)


if __name__ == "__main__":
    main()