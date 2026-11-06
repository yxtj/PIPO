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


def run_server(host: str, port: int, model: torch.nn.Module, inshape: tuple, n: int,
               model_name: str = None):
    # listening on port
    s = socket.create_server((host, port))
    print("Server is running on {}:{}".format(host, port))
    conn, addr = s.accept()
    print("Client connected from: {}".format(addr))
    # initialize server (parse the model, send info, build layers)
    t0 = time.time()
    server = Server(conn, model, inshape, model_name=model_name)
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