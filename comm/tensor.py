import socket
import struct
import io
import torch
from .basic import _recv_big_data_


def serialize_torch(data:torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    return buffer.read()

def deserialize_torch(data:bytes) -> torch.Tensor:
    buffer = io.BytesIO(data)
    result = torch.load(buffer)
    result.requires_grad = False
    return result

def send_torch(s:socket.socket, data:torch.Tensor) -> int:
    data = serialize_torch(data)
    n = len(data)
    # buffer = io.BytesIO()
    # torch.save(data, buffer)
    # n = buffer.tell()
    # buffer.seek(0)
    s.send(struct.pack('!i', n))
    # s.sendall(buffer.read())
    s.sendall(data)
    return 4 + n
    
def recv_torch(s:socket.socket, buf_sz:int=4096) -> tuple[torch.Tensor, int]:
    data = s.recv(4)
    nbytes, = struct.unpack('!i', data)
    buffer = _recv_big_data_(s, nbytes, buf_sz)
    # result = torch.load(io.BytesIO(buffer))
    # result.requires_grad = False
    result = deserialize_torch(buffer)
    return result, 4 + nbytes

