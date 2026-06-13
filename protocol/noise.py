from .ptobase import ProBaseServer, ProBaseClient, NumberType
from typing import Union

import socket
import torch
import time
import numpy as np
import comm
from Pyfhel import Pyfhel

from layer_basic.stat import Stat

__ALL__ = ['ProtocolClient', 'ProtocolServer']

class ProtocolClient(ProBaseClient):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.r = None
        self.pre = None

    def setup(self, ishape:tuple, oshape:tuple, r: NumberType=None, **kwargs) -> None:
        super().setup(ishape, oshape)
        self.r = self._gen_add_share_(r, ishape)

    def send_offline(self) -> None:
        self.basic_send_offline(self.r)
    
    def recv_offline(self) -> torch.Tensor:
        data = self.basic_recv_offline()
        self.pre = data
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        data = data - self.r
        self.stat.byte_online_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = data + self.pre
        return data


class ProtocolServer(ProBaseServer):
    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None,
              s: NumberType=None, m: NumberType=None, **kwargs) -> None:
        assert isinstance(last, ProtocolServer)
        assert 'noise_sigma' in kwargs, 'noise_sigma is required in kwargs'
        super().setup(ishape, oshape, last)
        self.mlast = last.m if last is not None else 1
        self.s = self._gen_add_share_(s, oshape)
        self.m = self._gen_mul_share_(m, oshape)
        sigma = kwargs['noise_sigma']
        assert isinstance(sigma, (int, float))
        self.noise_sigma = sigma
        
    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        t0 = time.time()
        data = self.basic_recv_offline()
        t1 = time.time()
        if not (isinstance(self.mlast, (int, float)) and self.mlast == 1):
            data /= self.mlast
        t2 = time.time()
        self.stat.time_offline_recv += t1 - t0
        self.stat.time_offline_comp += t2 - t1
        return data
    
    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        t0 = time.time()
        data *= self.m
        data += self.s
        t1 = time.time()
        self.basic_send_offline(data)
        t2 = time.time()
        self.stat.time_offline_comp += t1 - t0
        self.stat.time_offline_send += t2 - t1
    
    def recv_online(self) -> torch.Tensor:
        '''
        Receive (data/mlast) = (x - r/mlast) from client.
        '''
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        data = data/self.mlast
        t2 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        self.stat.time_online_comp += t2 - t1
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        '''
        Send (data*m - s + noise) to client. (data = W(x-r/mlast))
        '''
        t0 = time.time()
        data *= self.m
        data -= self.s
        data += torch.normal(0, self.noise_sigma, data.shape)
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1
