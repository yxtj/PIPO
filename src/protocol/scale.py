from .ptobase import ProBaseServer, ProBaseClient, NumberType
from typing import Union

import socket
import torch
import time
import numpy as np
import src.comm as comm
from src.heutil import HEEngine
from src.layer_basic.stat import Stat

__ALL__ = ['ProtocolClient', 'ProtocolServer']

class ProtocolClient(ProBaseClient):
    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.r = None
        self.pre = None
    
    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        super().setup(ishape, oshape)
        self.r = self._gen_add_share_(kwargs.get('r'), ishape)

    def send_offline(self) -> None:
        t = time.time()
        self.basic_send_offline(self.r)
        self.stat.time_offline_send += time.time() - t
    
    def recv_offline(self) -> torch.Tensor:
        t0 = time.time()
        data = self.basic_recv_offline()
        t1 = time.time()
        self.pre = data
        t2 = time.time()
        self.stat.time_offline_recv += t1 - t0
        self.stat.time_offline_comp += t2 - t1
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        t0 = time.time()
        data = data - self.r
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        data = data + self.pre
        t2 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        self.stat.time_online_comp += t2 - t1
        return data


class ProtocolServer(ProBaseServer):
    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.mlast = None
        self.s = None
        self.m = None
    
    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None, **kwargs) -> None:
        '''
        A local layer propagates the previous mask (reshaped by a flatten
        layer); the last non-local layer uses m = 1. Otherwise m is random
        unless supplied through kwargs by an internal sub-protocol.
        Similarly, s is random unless supplied through kwargs; local layers
        use s = 0.
        '''
        super().setup(ishape, oshape, last, **kwargs)
        self.mlast = last.m if last is not None else 1
        if self.is_local_layer:
            s = 0
            if self.ltype == 'flatten':
                m = last.m.flatten(1, -1) if last is not None else 1
            else:
                m = last.m if last is not None else 1
        else:
            s = kwargs.get('s')
            m = kwargs.get('m')
            if m is None and self.is_last_layer:
                m = 1.0
        self.s = self._gen_add_share_(s, oshape)
        self.m = self._gen_mul_share_(m, oshape)

    def gen_mpooling(self, stride_shape: tuple) -> ProBaseServer:
        block = torch.ones(stride_shape, device=self.device)
        mp = torch.kron(self.m, block) # kronecker product
        sp = torch.rand_like(mp)
        pro = ProtocolServer(self.socket, self.stat, self.he, self.device)
        pro.setup(self.ishape, mp.shape, last=self.last, s=sp, m=mp, share_ctx=False)
        return pro

    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        t0 = time.time()
        data = self.basic_recv_offline()
        t1 = time.time()
        if not (isinstance(self.mlast, (int, float)) and self.mlast == 1):
            # print(data.device, self.mlast.device)
            data /= self.mlast
            t2 = time.time()
            self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline_recv += t1 - t0
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
        Send (data*m + s) to client. (data = W(x-r/mlast))
        '''
        t0 = time.time()
        data *= self.m
        data -= self.s
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1
