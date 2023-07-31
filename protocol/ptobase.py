import socket
import torch
import numpy as np
from Pyfhel import Pyfhel
import struct
from typing import Union
import heutil

import comm
from layer_basic.stat import Stat
from .sshare import gen_add_share, gen_mul_share
from setting import USE_HE

NumberType = Union[int, float, torch.Tensor, np.ndarray]

# Protocol API: setup, send_online, recv_online, send_offline, recv_offline
# The send/recv methods should be state-free functions.

class ProtocolBase():
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str):
        self.socket = s
        self.stat = stat
        self.he = he
        self.device = device
        self.ishape = None
        self.oshape = None
        
    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        self.ishape = ishape
        self.oshape = oshape

    def send_online(self, data: torch.Tensor) -> None:
        raise NotImplementedError
    
    def recv_online(self) -> torch.Tensor:
        raise NotImplementedError

    # helpers for serializing and deserializing local state

    def to_bytes(self) -> bytes:
        return b''
    
    def from_bytes(self, b: bytes) -> None:
        return None
    
    # helpers for generating shares

    def _gen_add_share_(self, v: NumberType, shape: tuple) -> torch.Tensor:
        if v is not None:
            assert isinstance(v, (int, float, torch.Tensor, np.ndarray))
            if isinstance(v, torch.Tensor):
                assert v.shape == shape
                res = v
            elif isinstance(v, np.ndarray):
                assert v.shape == shape
                res = torch.from_numpy(v)
            elif isinstance(v, (int, float)):
                # res = v
                res = torch.zeros(shape, device=self.device).fill_(v)
            res = res.to(self.device)
        else:
            res = gen_add_share(shape, device=self.device)
        return res
    
    def _gen_mul_share_(self, v: NumberType, shape: tuple) -> torch.Tensor:
        if v is not None:
            assert isinstance(v, (int, float, torch.Tensor, np.ndarray))
            if isinstance(v, torch.Tensor):
                assert v.shape == shape
                res = v
            elif isinstance(v, np.ndarray):
                assert v.shape == shape
                res = torch.from_numpy(v)
            elif isinstance(v, (int, float)):
                # res = v
                if v == 0:
                    res = torch.zeros(shape, device=self.device)
                elif v == 1:
                    res = torch.ones(shape, device=self.device)
                else:
                    res= torch.zeros(shape, device=self.device).fill_(v)
            res = res.to(self.device)
        else:
            res = gen_mul_share(shape, device=self.device)
        return res


# Client workflow: send -> recv
class ProBaseClient(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
    
    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        super().setup(ishape, oshape)
        if USE_HE:
            b_ctx = self.he.to_bytes_context()
            self.socket.sendall(struct.pack('!i', len(b_ctx)) + b_ctx)
            self.stat.byte_offline_send += 4 + len(b_ctx)

    def send_offline(self) -> None:
        raise NotImplementedError
    
    def recv_offline(self) -> torch.Tensor:
        raise NotImplementedError
    
    def basic_send_offline(self, data: torch.Tensor) -> None:
        if USE_HE:
            # encrypt and send
            data = heutil.encrypt(self.he, data)
            nbyte = comm.send_he_matrix(self.socket, data, self.he)
        else:
            nbyte = comm.send_torch(self.socket, data)
        self.stat.byte_offline_send += nbyte
    
    def basic_recv_offline(self) -> torch.Tensor:
        if USE_HE:
            # recv and decrypt
            data, nbyte = comm.recv_he_matrix(self.socket, self.he)
            data = heutil.decrypt(self.he, data)
        else:
            data, nbyte = comm.recv_torch(self.socket)
            data = data.to(self.device)
        self.stat.byte_offline_recv += nbyte
        return data


# Server workflow: recv -> process -> send
class ProBaseServer(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.last = None
    
    def setup(self, ishape: tuple, oshape: tuple, last: ProtocolBase=None, **kwargs) -> None:
        super().setup(ishape, oshape)
        self.last = last
        if USE_HE:
            len = struct.unpack('!i', self.socket.recv(4))[0]
            b_ctx = self.socket.recv(len)
            self.he.from_bytes_context(b_ctx)
            self.stat.byte_offline_recv += 4 + len
    
    def setup_local(self, ishape: tuple, oshape: tuple, last: ProtocolBase=None,
        ltype: str='activation', **kwargs) -> None:
        assert ltype in ['activation', 'flatten', 'identity']
        raise NotImplementedError

    def gen_mpooling(self, stride_shape: tuple) -> ProtocolBase:
        raise NotImplementedError

    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        raise NotImplementedError
    
    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError

    def basic_send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        if USE_HE:
            # just send
            nbyte = comm.send_he_matrix(self.socket, data, self.he)
        else:
            nbyte = comm.send_torch(self.socket, data)
        self.stat.byte_offline_send += nbyte
    
    def basic_recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        if USE_HE:
            # just recv
            data, nbyte = comm.recv_he_matrix(self.socket, self.he)
        else:
            data, nbyte = comm.recv_torch(self.socket)
            data = data.to(self.device)
        self.stat.byte_offline_recv += nbyte
        return data
    