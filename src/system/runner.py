import time
import socket

import torch

from src.system import util, Server, Client
from src.system.resource import load_resource_set_server, load_resource_set_client, next_index, pick_index, save_resource_set_server, save_resource_set_client

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


def run_server(host: str, port: int, model: torch.nn.Module, inshape: tuple, n: int,
               cfg, do_offline: bool, do_online: bool, model_name: str = None):

    res_dir = cfg.common.resource.dir
    res_index = cfg.common.resource.index or None

    # listening on port
    s = socket.create_server((host, port))
    print("Server is running on {}:{}".format(host, port))
    conn, addr = s.accept()
    print("Client connected from: {}".format(addr))
    # initialize server (parse the model, send info, build layers)
    t0 = time.time()
    server = Server(conn, model, inshape, model_name=model_name)

    if do_offline:
        if do_online:
            # standard: run offline phase, resources are not persisted
            server.setup()
            print("Server is ready")
            server.offline()
        else:
            # offline-only: run offline phase, persist resources;
            # the new set index is announced to the client inside the model info
            index = next_index(res_dir, server.model_name or cfg.common.model.name, res_index)
            server.setup(resource=index)
            print("Server is ready (saving resource set {})".format(index))
            server.offline()
            save_resource_set_server(res_dir, server.model_name, index, server.info,
                                     [lyr.save_res() for lyr in server.layers])
    else:
        # online-only: restore resources instead of generating
        # server picks one and announces it to the client inside the model-info message
        index = pick_index(res_dir, server.model_name or cfg.common.model.name, res_index)
        server.setup(resource=index)
        entries = load_resource_set_server(res_dir, server.model_name, index, server.info)
        server.setup_layers(res=entries)
        print("Server is ready (resources restored, set {})".format(index))
    t1 = time.time()
    print("Server offline finished")
    # online phase
    if do_online:
        for i in range(n):
            server.online()
        print("Server online finished")
    t2 = time.time()
    conn.close()
    # finish
    if do_online:
        print("Quick measure: total offline time: {:.3f}; total online time: {:.3f}, average {}".format(
            t1 - t0, t2 - t1, (t2 - t1)/n))
    else:
        print("Quick measure: total offline time: {:.3f}".format(t1 - t0))
    print("Statistics with {} samples: ".format(n))
    show_stat(server.layers, n, do_offline, do_online)


def run_client(host: str, port: int, he, cfg, do_offline: bool, do_online: bool,
               dataset=None, n: int = 1, verify: bool = False):
    assert dataset is None or len(dataset) == n

    device = cfg.common.device
    res_dir = cfg.common.resource.dir
    res_index = cfg.common.resource.index or None
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
    if do_offline:
        if do_online:
            # standard: the server does not announce a resource set
            if client.resource is not None:
                s.close()
                raise RuntimeError(
                    "the server announced resource set {} (offline mode) "
                    "but the client runs in standard mode".format(client.resource))
            client.offline()
        else:
            # offline-only: run the offline phase, persist the generated resources
            # resources index is announced by the server
            index = client.resource
            if index is None:
                s.close()
                raise RuntimeError(
                    "the server did not announce a resource set: "
                    "the server must generate resources when the client does")
            if res_index is not None and res_index != index:
                s.close()
                raise RuntimeError(
                    "client requested resource set {} but the server announced {}".format(
                        res_index, index))
            client.offline()
            save_resource_set_client(res_dir, client.model_name, index, client.info,
                                     [lyr.save_res() for lyr in client.layers])
    else:
        # online-only: restore this side's resources; the server announced the set
        # index inside the model-info message, so both sides agree on it
        index = client.resource
        if index is None:
            s.close()
            raise RuntimeError(
                "the server did not announce a resource set: "
                "the server must run in online mode when the client does")
        if res_index is not None and res_index != index:
            s.close()
            raise RuntimeError(
                "client requested resource set {} but the server announced {}".format(
                    res_index, index))
        entries = load_resource_set_client(res_dir, client.model_name, index, client.info)
        client.setup_layers(res=entries)
        print("Client resources restored (set {})".format(index))
    t1 = time.time()
    print("Client offline finished")
    # online phase
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
        t2 = time.time()
        s.close()
        print("Client online finished")
        print("Quick measure: total offline time: {:.3f}; total online time: {:.3f}, average {}".format(
            t1 - t0, t2 - t1, (t2 - t1)/n))
    else:
        s.close()
        print("Quick measure: total offline time: {:.3f}".format(t1 - t0))
    print("Statistics with {} samples: ".format(n))
    show_stat(client.layers, n, do_offline, do_online)
    