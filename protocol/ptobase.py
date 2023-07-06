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

class ProtocolBase():
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        self.socket = s
        self.stat = stat
        self.he = he
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
                res = torch.zeros(shape).fill_(v)
        else:
            res = gen_add_share(shape)
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
                    res = torch.zeros(shape)
                elif v == 1:
                    res = torch.ones(shape)
                else:
                    res= torch.zeros(shape).fill_(v)
        else:
            res = gen_mul_share(shape)
        return res


# Client workflow: send -> recv
class ProBaseClient(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.r = None
        self.pre = None
    
    def setup(self, ishape:tuple, oshape:tuple, r: NumberType=None, **kwargs) -> None:
        super().setup(ishape, oshape)
        self.r = self._gen_add_share_(r, ishape)
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
        self.stat.byte_offline_recv += nbyte
        return data


# Server workflow: recv -> process -> send
class ProBaseServer(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel):
        super().__init__(s, stat, he)
        self.mlast = None
        self.s = None
        self.m = None
    
    def setup(self, ishape: tuple, oshape: tuple, s: NumberType=None, m: NumberType=None,
              last: ProtocolBase=None, **kwargs) -> None:
        '''
        if m is None, then generate m randomly (>0).
        if s is None, then generate s randomly.
        '''
        super().setup(ishape, oshape)
        mlast = last.m if last is not None else 1
        self.mlast = mlast
        self.s = self._gen_add_share_(s, oshape)
        self.m = self._gen_mul_share_(m, oshape)
        if USE_HE:
            len = struct.unpack('!i', self.socket.recv(4))[0]
            b_ctx = self.socket.recv(len)
            self.he.from_bytes_context(b_ctx)
            self.stat.byte_offline_recv += 4 + len
    
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
        self.stat.byte_offline_recv += nbyte
        return data
    