import sys
import time
import torch
import socket
from Pyfhel import Pyfhel

from system import util, Server, Client


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
    s_relu.show_offline("  ReLU")
    s_linear.show_offline("  Linear")
    s_l_conv.show_offline("  Linear-Conv")
    s_l_fc.show_offline("  Linear-FC")
    s_sc.show_offline("  Shortcut")
    s_pool.show_offline("  Pool")
    print()
    s_total.show_online("Total", n)
    s_relu.show_online("  ReLU", n)
    s_linear.show_online("  Linear", n)
    s_l_conv.show_online("  Linear-Conv", n)
    s_l_fc.show_online("  Linear-FC", n)
    s_sc.show_online("  Shortcut", n)
    s_pool.show_online("  Pool", n)


def run_server(host: str, port: int, model: torch.nn.Module, inshape: tuple, n: int):
    # listening on port
    s = socket.create_server((host, port))
    print("Server is running on {}:{}".format(host, port))
    conn, addr = s.accept()
    print("Client connected from: {}".format(addr))
    # initialize server
    t0 = time.time()
    server = Server(conn, model, inshape)
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


def run_client(host: str, port: int, model: torch.nn.Module, inshape: tuple, he:Pyfhel,
               dataset=None, n: int=1, verify: bool=False):
    assert dataset is None or len(dataset) == n
    # connect to server
    s = socket.create_connection((host, port))
    print("Client is connecting to {}:{}".format(host, port))
    # initialize client
    t0 = time.time()
    client = Client(s, model, inshape, he)
    print("Client is ready")
    # offline phase
    client.offline()
    t1 = time.time()
    print("Client offline finished")
    # online phase
    if len(inshape) == 3:
        inshape = (1, *inshape)
    for i in range(n):
        d = torch.rand(inshape) if dataset is None else dataset[i]
        if next(model.parameters()).is_cuda:
            d = d.cuda()
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

