import sys
import torch
from Pyfhel import Pyfhel

from model import minionn
from system.runner import run_client, run_server
from setting import USE_HE

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print('Usage: python server|client [n=1] [device=cpu] [weight_file] [host=localhost] [port=8100]')
        sys.exit(1)
    # model_name = argv[1]
    mode = argv[1]
    assert mode in ['server', 'client']
    n = int(argv[2]) if len(argv) > 2 else 1
    device = argv[3] if len(argv) > 3 else 'cpu'
    assert device == 'cpu' or device.startswith('cuda')
    wfile = argv[4] if len(argv) > 4 else None
    host = argv[5] if len(argv) > 5 else 'localhost'
    port = int(argv[6]) if len(argv) > 6 else 8100
    
    # set model and inshape
    inshape = minionn.inshape
    model = minionn.build()
    if wfile is not None:
        model.load_state_dict(torch.load(wfile))
    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.to(device)
    print("Model loaded on {}".format(device))
    
    if mode == 'server':
        run_server(host, port, model, inshape, n)
    else:
        if USE_HE:
            he = Pyfhel()
            he.contextGen(scheme='ckks', n=2**13, scale=2**30, qi_sizes=[30]*5)
            he.keyGen()
        else:
            he = None
        run_client(host, port, model, inshape, he, None, n, True)
