from comm import ot
import sys
import socket
import time
import numpy as np
import multiprocessing as mp
from comm.util import serialize_numpy_meta, deserialize_numpy_meta

host, port = 'localhost', 12345

type_str = np.int32

# KEY: no meta for the indicator, because it may be XOR to meaningless data
def msg_gen_func(x0:np.ndarray, x1:np.ndarray, indicator:bytes):
    assert x0.shape == x1.shape
    flag = np.frombuffer(indicator, dtype=np.bool).reshape(x0.shape)
    r = x0*flag + x1*(~flag)
    return r.tobytes()


def run_server(nbits, m0, m1, n):
    prefix = '[Server]'
    sock = socket.create_server((host, port))
    conn, addr = sock.accept()
    print(f"{prefix} connection from {addr}")
    server = ot.ObliviousTransferReceiver(conn, nbits)
    print(f"{prefix} setup")
    server.setup()
    for i in range(n):
        flag = np.random.randint(0, 2, size=m0.shape, dtype=np.bool)
        print(prefix, f"Run {i}:")
        t = time.time()
        # KEY: no meta for the flag, because it may be XOR to meaningless data
        data, n_send, n_recv = server.run_customized(flag.tobytes())
        r = np.frombuffer(data, dtype=type_str).reshape(m0.shape)
        t = time.time() - t
        ref = m0*flag + m1*(~flag)
        check = np.all(r == ref)
        print(f"{prefix} send: {n_send} recv: {n_recv} time: {t} correct: {check}")
        # print(f"{prefix} recv={r}")


def run_client(nbits, m0, m1, n):
    prefix = '[Client]'
    sock = socket.create_connection((host, port))
    print(f"{prefix} connected to server")
    client = ot.ObliviousTransferSender(sock, nbits)
    print(f"{prefix} setup")
    print(f"{prefix} data shape: {m0.shape}")
    client.setup()
    for i in range(n):
        print(prefix, f"Run {i}")
        t = time.time()
        n_send, n_recv = client.run_customized(m0, m1, msg_gen_func)
        t = time.time() - t
        print(f"{prefix} send: {n_send} recv: {n_recv} time: {t}")


if __name__ == '__main__':
    # parse args
    args = sys.argv[1:]
    if len(args) > 3:
        print('usage: python ot_c.py nbits=1024 max_size=4 n=1')
        sys.exit(1)
    # mode = args[0]
    nbits = int(args[0]) if len(args) > 0 else 1024
    max_size = int(args[1]) if len(args) > 1 else 4
    n = int(args[2]) if len(args) > 2 else 1
    # set the data matrix size
    m0 = np.zeros((max_size, max_size), dtype=type_str) - 1
    m1 = np.zeros((max_size, max_size), dtype=type_str) + 1
    # run
    ps = mp.Process(target=run_server, args=(nbits, m0, m1, n))
    ps.start()
    pc = mp.Process(target=run_client, args=(nbits, m0, m1, n))
    pc.start()
    ps.join()
    pc.join()
