import socket
import numpy as np
import struct
from Pyfhel import Pyfhel, PyCtxt

from .basic import send_chunk, recv_chunk, _recv_big_data_
from .ndarray import serialize_numpy_meta, deserialize_numpy_meta


# HE ciphertext

def send_ciphertext(s:socket.socket, data:PyCtxt) -> int:
    return send_chunk(s, data.to_bytes())

def recv_ciphertext(s:socket.socket, he:Pyfhel, buf_sz:int=4096) -> tuple[PyCtxt, int]:
    data, n_recv = recv_chunk(s, buf_sz)
    res = PyCtxt(pyfhel=he, bytestring=data)
    return res, n_recv


def send_he_matrix(s:socket.socket, data:np.ndarray, he:Pyfhel) -> int:
    b_ctx = he.to_bytes_context()
    meta = serialize_numpy_meta(data)
    d = data.flatten()
    header = struct.pack('!i', len(b_ctx)) + b_ctx + meta + struct.pack('!i', len(d[0].to_bytes()))
    s.sendall(header)
    n = 4 + len(b_ctx) + len(meta) + 4
    for i in range(data.size):
        n += s.sendall(d[i].to_bytes())
    return n

def recv_he_matrix(s:socket.socket, he:Pyfhel, buf_sz:int=4096) -> tuple[np.ndarray, int]:
    header_len = struct.unpack('!i', s.recv(4))[0]
    data = s.recv(header_len)
    # recv_chunk(s, buf_sz)
    ctx_len = struct.unpack('!i', data[:4])[0]
    ctx = data[4:4+ctx_len]
    meta_len, nbytes, type_str, shape = deserialize_numpy_meta(data[4+ctx_len:])
    ct_len = struct.unpack('!i', data[4+ctx_len+meta_len:])[0]
    # update HE context
    he.from_bytes_context(ctx)
    # parse ciphertexts
    res = np.empty(shape, dtype=object)
    r = res.flatten()
    size = np.prod(shape)
    for i in range(size):
        b = _recv_big_data_(s, ct_len, buf_sz)
        ct = PyCtxt(pyfhel=he, bytestring=b)
        r[i] = ct
    return res, 4 + header_len + nbytes
