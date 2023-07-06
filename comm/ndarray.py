import socket
import struct
import numpy as np

from comm.basic import _recv_big_data_

# numpy meta

def serialize_numpy_meta(data:np.ndarray) -> bytes:
    type_char = data.dtype.char.encode()
    shape = data.shape
    r = struct.pack('!ich', data.nbytes, type_char, len(shape)) + \
        struct.pack('!'+'i'*len(shape), *shape)
    return r

def deserialize_numpy_meta_phase1(data:bytes) -> tuple[int, int, str]:
    nbytes, type_char, shape_len = struct.unpack('!ich', data[:7])
    return nbytes, type_char, shape_len

def deserialize_numpy_meta_phase2(data:bytes, shape_len:int) -> tuple[int]:
    shape = struct.unpack('!'+'i'*shape_len, data[:shape_len*4])
    return shape

def deserialize_numpy_meta(data:bytes) -> tuple[int, int, str, tuple[int]]:
    nbytes, type_char, shape_len = deserialize_numpy_meta_phase1(data)
    header_len = 7 + shape_len*4
    shape = deserialize_numpy_meta_phase2(data[7:header_len], shape_len)
    return header_len, nbytes, type_char, shape


def serialize_numpy(data:np.ndarray) -> bytes:
    return serialize_numpy_meta(data) + data.tobytes()

def deserialize_numpy(data:bytes) -> np.ndarray:
    header_len, nbytes, type_char, shape = deserialize_numpy_meta(data)
    buffer = data[header_len:header_len+nbytes]
    result = np.frombuffer(buffer, dtype=type_char).reshape(shape)
    return result

# numpy
    
def send_numpy(s:socket.socket, data:np.ndarray) -> int:
    b_meta = serialize_numpy_meta(data)
    s.sendall(b_meta)
    data = data.tobytes()
    s.sendall(data)
    return len(b_meta) + len(data)
    
def recv_numpy(s:socket.socket, buf_sz:int=4096) -> tuple[np.ndarray, int]:
    data = s.recv(7)
    nbytes, type_char, shape_len = deserialize_numpy_meta_phase1(data)
    header_len = 7 + shape_len*4
    data = s.recv(shape_len*4)
    shape = deserialize_numpy_meta_phase2(data, shape_len)
    buffer = _recv_big_data_(s, nbytes, buf_sz)
    result = np.frombuffer(buffer, dtype=type_char).reshape(shape)
    return result, header_len + nbytes
