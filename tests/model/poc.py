import sys
import time
import torch
import socket
from Pyfhel import Pyfhel

from model import poc
import system


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 3:
        print('Usage: python server|client model [seed=0] [host=localhost] [port=8100]')
        sys.exit(1)
    mode = argv[1]
    assert mode in ['server', 'client']
    model_name = argv[2]
    seed = int(argv[3]) if len(argv) > 3 else 0
    host = argv[4] if len(argv) > 4 else 'localhost'
    port = int(argv[5]) if len(argv) > 5 else 8100
    
    # set seed to make sure the model is the same on client and server
    if seed is not None:
        torch.manual_seed(seed)
    
    # set model and inshape
    if model_name in poc.map:
        inshape, model = poc.map[model_name]
        model.eval()
        for p in model.parameters():
            p.data.uniform_(-1, 1)
        # for i, lyr in enumerate(model.modules()):
        #     for p in lyr.parameters():
        #         p.data.fill_(i + 1)
    else:
        print("Unknown model name: {}".format(model_name))
        print("Available models: {}".format(poc.map.keys()))
        sys.exit(1)
    
    if mode == 'server':
        s = socket.create_server((host, port))
        print("Server is running on {}:{}".format(host, port))
        conn, addr = s.accept()
        print("Client connected from: {}".format(addr))
        t0 = time.time()
        server = system.Server(conn, model, inshape)
        print("Server is ready")
        server.offline()
        t1 = time.time()
        print("Server offline finished")
        server.online()
        t2 = time.time()
        print("Server online finished")
        print("Statistics: ")
        print("Offline time: {:.2f}; Online time: {:.2f}".format(t1 - t0, t2 - t1))
        for i, lyr in enumerate(server.layers):
            print("  Layer {} {}: {}".format(i, lyr.__class__.__name__, lyr.stat))
    else:
        he = Pyfhel()
        he.contextGen(scheme='ckks', n=2**13, scale=2**30, qi_sizes=[30]*5)
        he.keyGen()
        s = socket.create_connection((host, port))
        print("Client is connecting to {}:{}".format(host, port))
        t0 = time.time()
        client = system.Client(s, model, inshape, he)
        print("Client is ready")
        client.offline()
        t1 = time.time()
        print("Client offline finished")
        inshape = (1, *inshape)
        data = torch.rand(inshape)
        # data = torch.arange(1, 1 + torch.prod(torch.tensor(inshape)).item(), dtype=torch.float).view(inshape)
        with torch.no_grad():
            res = client.online(data)
        t2 = time.time()
        print("Client online finished")
        print("System result: {}".format(res))
        with torch.no_grad():
            res2 = model(data)
        print("Local result: {}".format(res2))
        diff = (res - res2).abs().sum()
        print("Difference: {}".format(diff))
        print("Statistics: ")
        print("Offline time: {:.2f}; Online time: {:.2f}".format(t1 - t0, t2 - t1))
        for i, lyr in enumerate(client.layers):
            print("  Layer {} {}: {}".format(i, lyr.__class__.__name__, lyr.stat))
