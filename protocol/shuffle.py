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
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.mlast = None
        self.rplast = None
        self.s = None
        self.m = None
        self.p = None
        self.rp = None
        self.offline_buffer = None # used to generate random data for offline phase
    
    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None,
            s: NumberType=None, m: NumberType=None, p=None, **kwargs) -> None:
        '''
        If m is None, then generate m randomly (>0).
        If s is None, then generate s randomly.
        If p is None, then generate p randomly. If p is 1, then no shuffle.
        '''
        assert last is None or isinstance(last, ProtocolServer)
        super().setup(ishape, oshape, last)
        assert p is None or isinstance(p, int) or p.shape == ishape
        mlast = last.m if last is not None else 1
        self.mlast = mlast
        if last is None:
            self.rplast = 1
        else:
            self.rplast = torch.argsort(last.p.ravel()).reshape(self.ishape)
        self.s = self._gen_add_share_(s, oshape)
        self.m = self._gen_mul_share_(m, oshape)
        if p is None:
            n = np.prod(self.oshape)
            self.p = torch.randperm(n).reshape(self.oshape) # shuffle matrix
        else:
            self.p = 1
        # self.rp = torch.argsort(self.p.ravel()).reshape(self.ishape) # unshuffle matrix
    
    def setup_local(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None, 
            ltype: str='activation',  **kwargs) -> None:
        if ltype == 'flatten':
            m = last.m.flatten(1, -1) if last is not None else 1
        else:
            m = last.m if last is not None else 1
        s = 0
        p = 1
        self.setup(ishape, oshape, last, s, m, p)

    def gen_mpooling(self, stride_shape: tuple) -> ProBaseServer:
        block = torch.ones(stride_shape, device=self.device)
        mp = torch.kron(self.m, block) # kronecker product
        sp = torch.rand_like(mp)
        # TODO: block-wise shuffle
        self.p = 1
        pp = 1
        pro = ProtocolServer(self.socket, self.stat, self.he, self.device)
        pro.setup(self.ishape, mp.shape, last=self.last, s=sp, m=mp, p=pp)
        return pro
        
    def deshuflle_input(self, data: torch.Tensor) -> torch.Tensor:
        if isinstance(self.rplast, int):
            res = data
        else:
            res = data.ravel()[self.rplast].reshape(self.ishape)
        return res

    def shuffle_output(self, data: torch.Tensor) -> torch.Tensor:
        if isinstance(self.p, int):
            res = data
        else:
            res = data.ravel()[self.p].reshape(self.oshape)
        return res
    
    def recv_offline(self) -> torch.Tensor:
        t0 = time.time()
        data = self.basic_recv_offline()
        t1 = time.time()
        data = self.deshuflle_input(data)
        if not (isinstance(self.mlast, (int, float)) and self.mlast == 1):
            # print(data.device, self.mlast.device)
            data /= self.mlast
        t2 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline_recv += t1 - t0
        return data
        
    def send_offline(self, data: np.ndarray) -> None:
        t0 = time.time()
        data *= self.m
        data = self.shuffle_output(data)
        data += self.s
        t1 = time.time()
        self.basic_send_offline(data)
        t2 = time.time()
        self.stat.time_offline_comp += t1 - t0
        self.stat.time_offline_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        '''
        Receive plast(x*mlast - r) from client.
        Return x - r/mlast.
        '''
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        data = self.deshuflle_input(data)
        data /= self.mlast
        t2 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        self.stat.time_online_comp += t2 - t1
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        '''
        Input data = W(x-r/mlast)
        Send ( p(data*m) + s ) to client.
        '''
        t0 = time.time()
        data *= self.m
        data = self.shuffle_output(data)
        data -= self.s
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1
