"""PIPO launcher: `python -m src.main server|client [options]`.

Replaces the old example/ scripts: the picked model and all launch parameters
come from config.toml / CLI arguments.
"""

import argparse
import sys

from src.common.config import apply_args, load_config, set_config

_AVAILABLE_PROTOCOL = ("plaintext", "scale", "shuffle", "noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipo", description="PIPO privacy-preserving inference")
    parser.add_argument("mode", choices=["server", "client"], help="run as server or client")
    parser.add_argument("--config", default=None, help="config file (default: config.toml)")
    for flag, dest in [
        ("--host", "host"), ("--port", "port"), ("--device", "device"),
        ("--model", "model"), ("--n", "n"), ("--seed", "seed"),
        ("--wfile", "wfile"), ("--protocol", "protocol"),
    ]:
        converter = int if dest in ("port", "n", "seed") else None
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
    mapping = {f"common.{k}": k for k in
               ("host", "port", "device", "model", "n", "seed", "wfile", "protocol", "use_he")}
    mapping["client.verify"] = "verify"
    cfg = apply_args(cfg, args, mapping)
    common = cfg.common
    assert common.protocol in _AVAILABLE_PROTOCOL, f"Invalid protocol: {common.protocol}. Available: {_AVAILABLE_PROTOCOL}"
    set_config(cfg)

    # late imports so that the protocol variant is chosen from the final config
    from src.models import registry

    inshape, model = registry.load(common.model, seed=common.seed, wfile=common.wfile or None, device=common.device)
    print("Model loaded: {} on {}".format(common.model, common.device))

    try:  # Pyfhel is optional; only needed with use_he
        from Pyfhel import Pyfhel
        has_pyfhel = True
    except ImportError:
        has_pyfhel = False
    if common.use_he:
        if not has_pyfhel:
            print("Error: use_he is enabled but Pyfhel is not installed.")
            sys.exit(1)
        he = Pyfhel()
        he.contextGen(scheme="ckks", n=2**13, scale=2**30, qi_sizes=[30]*5)
        he.keyGen()
    else:
        he = None

    if args.mode == "server":
        from src.system.runner import run_server
        run_server(common.host, common.port, model, inshape, common.n)
    else:
        from src.system.runner import run_client
        run_client(common.host, common.port, model, inshape, he,
                   dataset=None, n=common.n, verify=cfg.client.verify)


if __name__ == "__main__":
    main()
