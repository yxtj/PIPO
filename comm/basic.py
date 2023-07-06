import socket
import struct


def _recv_big_data_(s:socket.socket, n:int, buf_sz:int=4096) -> bytes:
    buffer = []
    while n > 0:
        t = s.recv(min(buf_sz, n))
        n -= len(t)
        buffer.append(t)
    data = b''.join(buffer)
    return data

# byte chunk

def send_chunk(s:socket.socket, data:bytes) -> int:
    s.sendall(struct.pack('!i', len(data)))
    s.sendall(data)
    return 4 + len(data)

def recv_chunk(s:socket.socket, buf_sz:int=4096) -> tuple[bytes, int]:
    data = s.recv(4)
    n, = struct.unpack('!i', data)
    return _recv_big_data_(s, n, buf_sz), 4 + n

# shape tuple

def send_shape(s:socket.socket, shape:tuple[int]):
    s.sendall(struct.pack('!i', len(shape)))
    s.sendall(struct.pack('!'+'i'*len(shape), *shape))
    
def recv_shape(s:socket.socket) -> tuple[int]:
    data = s.recv(4)
    n, = struct.unpack('!i', data)
    data = s.recv(n*4)
    shape = struct.unpack('!'+'i'*n, data)
    return shape

