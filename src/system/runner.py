import sys
import time
import torch
import socket
from typing import Union

from src.system import util, Server, Client


def show_stat(layers, n):
    s_total, s_relu, s_linear, s_l_conv, s_l_fc, s_pool, s_sc = util.analyze_stat(layers, n)
    # print()
    # s_total.show("Total", n)
    # s_relu.show("  ReLU", n)
    # s_linear.show("  Linear", n)
    # s_l_conv.show("  Linear-Conv", n)
    # s_l_fc.show("  Linear-FC", n)
    # s_sc.show("  Shortcut", n)
    # s_pool.show("  Pool", n)
    print()
    s_total.show_offline("Total")
    s_relu.show_offline("  Activation")
    s_linear.show_offline("  Linear")
    s_l_conv.show_offline("  Linear-Conv")
    s_l_fc.show_offline("  Linear-FC")
    s_sc.show_offline("  Shortcut")
    s_pool.show_offline("  Pool")
    print()
    s_total.show_online("Total", n)
    s_relu.show_online("  Activation", n)
    s_linear.show_online("  Linear", n)
    s_l_conv.show_online("  Linear-Conv", n)
    s_l_fc.show_online("  Linear-FC", n)
    s_sc.show_online("  Shortcut", n)
    s_pool.show_online("  Pool", n)


def run_offline(model: torch.nn.Module, inshape: tuple, model_name: str = None,
                res_dir: str = None, res_index: int = None, res_count: int = 1):
    """Run mode 'offline': a single process plays both roles over a socket
    pair, runs the full interactive offline phase and persists one resource
    set (one server + one client file) per round."""
    import threading

    from src.common.config import get_config, he_engine_args
    from src.heutil import create_he
    from src.system.offline_res import next_index, save_res_set

    cfg = get_config()
    res_dir = res_dir or cfg.common.offline.dir
    # the client side owns the keyed engine (same as a normal client run);
    # its HE context reaches the server side through the socket pair
    backend, params = he_engine_args(cfg)
    he = create_he(backend, params=params, init_keys=True)

    index = next_index(res_dir, model_name or cfg.common.model.name, res_index)
    for k in range(res_count):
        cur = index + k
        print("=== generating offline resource set {} ===".format(cur))
        srv_sock, cli_sock = socket.socketpair()
        server = Server(srv_sock, model, inshape, model_name=model_name)
        client = Client(cli_sock, he=he)
        server.setup()
        client.setup()
        # the offline round is an interactive exchange: run each role in its
        # own thread so the send/recv pairs can interleave
        t = threading.Thread(target=server.offline, daemon=True)
        t.start()
        client.offline()
        t.join()
        server_entries = [lyr.save_res() for lyr in server.layers]
        client_entries = [lyr.save_res() for lyr in client.layers]
        save_res_set(res_dir, server.model_name, cur, server.info,
                     server_entries, client_entries)
        srv_sock.close()
        cli_sock.close()


def run_server(host: str, port: int, model: torch.nn.Module, inshape: tuple, n: int,
               model_name: str = None):
    from src.common.config import get_config

    cfg = get_config()
    # listening on port
    s = socket.create_server((host, port))
    print("Server is running on {}:{}".format(host, port))
    conn, addr = s.accept()
    print("Client connected from: {}".format(addr))
    # initialize server (parse the model, send info, build layers)
    t0 = time.time()
    server = Server(conn, model, inshape, model_name=model_name)
    if cfg.common.mode == "online":
        # restore precomputed offline resources instead of interacting: the
        # server picks the set (explicit --res-index or a random one) and
        # announces it to the client inside the model-info message
        from src.system.offline_res import load_res_set, pick_index

        index = pick_index(cfg.common.offline.dir,
                           server.model_name or cfg.common.model.name,
                           cfg.common.offline.index or None)
        server.setup(offline_res=index)
        entries = load_res_set(cfg.common.offline.dir, server.model_name, index,
                               "server", server.info)
        server.setup_layers(res=entries)
        print("Server is ready (offline resources restored, set {})".format(index))
    else:
        server.setup()
        print("Server is ready")
        # offline phase
        server.offline()
    t1 = time.time()
    print("Server offline finished")
    # online phase
    for i in range(n):
        server.online()
    t2 = time.time()
    conn.close()
    # finish
    print("Server online finished")
    print("Quick measure: total offline time: {:.3f}; total online time: {:.3f}, average {}".format(t1 - t0, t2 - t1, (t2 - t1)/n))
    print("Statistics with {} samples: ".format(n))
    show_stat(server.layers, n)


def run_client(host: str, port: int, he, dataset=None, n: int=1, verify: bool=False):
    assert dataset is None or len(dataset) == n

    from src.common.config import get_config

    cfg = get_config()
    device = cfg.common.device
    # connect to server
    s = socket.create_connection((host, port))
    print("Client is connecting to {}:{}".format(host, port))
    # initialize client (receive the server-parsed model info)
    t0 = time.time()
    client = Client(s, he=he)
    client.setup()
    print("Client is ready")
    inshape = client.inshape

    # verify needs a local model only as the plaintext reference; the layers
    # used for inference always come from the server-sent model info
    model = None
    if verify:
        from models import registry
        from src.common.model_info import build_model_info
        from src.common.model_info_io import model_info_hash

        try:
            model_cfg = cfg.common.model
            _, model = registry.load(
                client.model_name,
                seed=model_cfg.seed,
                wfile=model_cfg.wfile or None,
                wfolder=model_cfg.wfolder or None,
                device=device,
            )
            local_info = build_model_info(model, client.info.inshape)
            server_hash = model_info_hash(client.info)
            local_hash = model_info_hash(local_info)
            if local_hash != server_hash:
                raise RuntimeError(
                    "client model {!r} structure differs from the server model "
                    "(local hash {}, server hash {}); verify requires identical "
                    "model definitions".format(
                        client.model_name, local_hash[:12], server_hash[:12]))
        except BaseException:
            s.close()
            raise
        print("Verify: client model matches server model info (hash {})".format(server_hash[:12]))

    # offline phase
    if cfg.common.mode == "online":
        # restore precomputed offline resources: the server announced the set
        # index inside the model-info message; both sides must agree on it
        from src.system.offline_res import load_res_set

        index = client.offline_res
        if index is None:
            s.close()
            raise RuntimeError(
                "the server did not announce an offline resource set: "
                "the server must run in online mode when the client does")
        want = cfg.common.offline.index or None
        if want is not None and want != index:
            s.close()
            raise RuntimeError(
                "client requested offline resource set {} but the server announced {}".format(
                    want, index))
        entries = load_res_set(cfg.common.offline.dir, client.model_name, index,
                               "client", client.info)
        client.setup_layers(res=entries)
        print("Client offline resources restored (set {})".format(index))
    else:
        if client.offline_res is not None:
            s.close()
            raise RuntimeError(
                "the server announced offline resource set {} (online mode) "
                "but the client runs in {} mode".format(client.offline_res, cfg.common.mode))
        client.offline()
    t1 = time.time()
    print("Client offline finished")
    # online phase
    input_shape = inshape
    if len(inshape) == 3 or len(inshape) == 1:
        input_shape = (1, *inshape)
    for i in range(n):
        d = torch.rand(input_shape) if dataset is None else dataset[i]
        d = d.to(device)
        with torch.no_grad():
            res = client.online(d)
        if verify:
            with torch.no_grad():
                res2 = model(d)
            diff = torch.abs(res - res2)
            print("Verify {}: mean absolute difference: {:.6g}, mean relative difference: {:.6g}".format(
                i, diff.mean(), torch.nanmean(diff/res2)))
    t2 = time.time()
    s.close()
    # finish
    print("Client online finished")
    print("Quick measure: total offline time: {:.3f}; total online time: {:.3f}, average {}".format(t1 - t0, t2 - t1, (t2 - t1)/n))
    print("Statistics with {} samples: ".format(n))
    show_stat(client.layers, n)