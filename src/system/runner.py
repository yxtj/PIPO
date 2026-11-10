import time
import socket

import torch

from src.system import util, Server, Client
from src.system.resource import (next_index, save_resource_set_client,
                                 save_resource_set_server)


def show_stat(layers, n, do_offline=True, do_online=True):
    s_total, s_relu, s_linear, s_l_conv, s_l_fc, s_pool, s_sc = util.analyze_stat(layers, n)
    if do_offline:
        print()
        s_total.show_offline("Total")
        s_relu.show_offline("  Activation")
        s_linear.show_offline("  Linear")
        s_l_conv.show_offline("  Linear-Conv")
        s_l_fc.show_offline("  Linear-FC")
        s_sc.show_offline("  Shortcut")
        s_pool.show_offline("  Pool")
    if do_online:
        print()
        s_total.show_online("Total", n)
        s_relu.show_online("  Activation", n)
        s_linear.show_online("  Linear", n)
        s_l_conv.show_online("  Linear-Conv", n)
        s_l_fc.show_online("  Linear-FC", n)
        s_sc.show_online("  Shortcut", n)
        s_pool.show_online("  Pool", n)


def _layer_res(layers):
    """Snapshot the per-layer offline resources of one side."""
    return [lyr.save_res() for lyr in layers]

def _finish(layers, n, do_offline, do_online, t0, t1, t2):
    """Shared timing/statistics report for both roles."""
    if do_online:
        print("Quick measure: total offline time: {:.3f}; total online time: {:.3f}, average {}".format(
            t1 - t0, t2 - t1, (t2 - t1)/n))
    else:
        print("Quick measure: total offline time: {:.3f}".format(t1 - t0))
    kind = "resource sets" if (do_offline and not do_online) else "samples"
    print("Statistics with {} {}: ".format(n, kind))
    show_stat(layers, n, do_offline, do_online)


def run_server(host: str, port: int, model: torch.nn.Module, inshape: tuple, n: int,
               cfg, do_offline: bool, do_online: bool, model_name: str = None):

    res_dir = cfg.common.resource.dir
    res_index = cfg.common.resource.index or None

    # listening on port
    s = socket.create_server((host, port))
    print("Server is running on {}:{}".format(host, port))
    conn, addr = s.accept()
    print("Client connected from: {}".format(addr))
    t0 = time.time()
    server = Server(conn, model, inshape, model_name=model_name)
    server.setup()
    print("Server is ready")

    if do_offline:
        # offline/standard: interactive offline phase. In offline-only mode n
        # resource sets are generated and persisted, starting from the next
        # free id (both sides compute it locally from the shared directory
        # before the first round, so they agree on the ids).
        start = None if do_online else next_index(res_dir, server.model_name, res_index)
        for k in range(n):
            server.offline()
            if start is not None:
                save_resource_set_server(res_dir, server.model_name, start + k,
                                         server.info, _layer_res(server.layers))
    else:
        # online-only: negotiate the set id with the client and restore it
        index = server.select_resource(res_dir, res_index)
        server.restore(res_dir, index)
    t1 = time.time()
    print("Server offline finished")

    if do_online:
        for _ in range(n):
                server.online()
        print("Server online finished")
    t2 = time.time()
    conn.close()
    _finish(server.layers, n, do_offline, do_online, t0, t1, t2)


def run_client(host: str, port: int, he, cfg, do_offline: bool, do_online: bool,
               dataset=None, n: int = 1, verify: bool = False):
    assert dataset is None or len(dataset) == n

    device = cfg.common.device
    res_dir = cfg.common.resource.dir
    res_index = cfg.common.resource.index or None
    # connect to server
    s = socket.create_connection((host, port))
    print("Client is connecting to {}:{}".format(host, port))
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

    if do_offline:
        # offline/standard: interactive offline phase. In offline-only mode the
        # client persists its half of n resource sets under the same ids the
        # server derives (see run_server).
        start = None if do_online else next_index(res_dir, client.model_name, res_index)
        for k in range(n):
            client.offline()
            if start is not None:
                save_resource_set_client(res_dir, client.model_name, start + k,
                                         client.info, _layer_res(client.layers))
    else:
        # online-only: negotiate the set id with the server and restore it
        index = client.select_resource(res_index)
        client.restore(res_dir, index)
    t1 = time.time()
    print("Client offline finished")

    if do_online:
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
        print("Client online finished")
    t2 = time.time()
    s.close()
    _finish(client.layers, n, do_offline, do_online, t0, t1, t2)
