from .ptobase import ProBaseServer, ProBaseClient, NumberType
from .crypto import block_shuffle_perm, gen_shuffle, gen_reverse_shuffle, shuffle_tensor, deshuffle_tensor
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
    res_fields = ('r', 'pre')

    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.r = None
        self.pre = None
    
    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        super().setup(ishape, oshape, **kwargs)
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
    res_fields = ('s', 'm', 'p')

    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.mlast = None
        self.rplast = None
        self.s = None
        self.m = None
        self.p = None
        self.rp = None
        self.offline_buffer = None # used to generate random data for offline phase
    
    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None, **kwargs) -> None:
        '''
        A local layer propagates the previous masks (m reshaped by a flatten
        layer); the last non-local layer uses m = 1 and the identity
        permutation. Otherwise m/p are random unless supplied through kwargs
        by an internal sub-protocol.
        '''
        super().setup(ishape, oshape, last, **kwargs)
        assert last is None or isinstance(last, ProtocolServer)
        self.mlast = last.m if last is not None else 1
        if self.is_local_layer:
            s = 0
            if self.ltype == 'flatten':
                m = last.m.flatten(1, -1) if last is not None else 1
            else:
                m = last.m if last is not None else 1
            p = last.p if last is not None else 1
        else:
            s = kwargs.get('s')
            m = kwargs.get('m')
            p = kwargs.get('p')
            if self.is_last_layer:
                if m is None:
                    m = 1.0
                if p is None:
                    p = 1
        assert p is None or isinstance(p, int) or p.numel() == int(np.prod(oshape))
        if last is None or isinstance(last.p, int):
            self.rplast = 1
        else:
            self.rplast = gen_reverse_shuffle(last.p)
        self.s = self._gen_add_share_(s, oshape)
        self.m = self._gen_mul_share_(m, oshape)
        if p is None:
            self.p = gen_shuffle(self.oshape, device=self.device) # shuffle matrix
        else:
            self.p = p
        # self.rp = torch.argsort(self.p.ravel()).reshape(self.ishape) # unshuffle matrix

    def gen_mpooling(self, stride_shape: tuple, pool_res: dict = None) -> ProBaseServer:
        # Block-wise shuffle (same policy as the shuffle protocol): the
        # previous layer's full permutation is undone first
        # (protocol_pool.rplast), then a block-wise one is applied to the
        # masked tensor the client pools on. Because mp = kron(m, ones) is
        # constant inside each pooling window, maxpool(p_full(x*mp)) ==
        # p_block(maxpool(x)*m): the client-held pooled tensor carries exactly
        # the pending permutation p_block and the multiplicative mask m this
        # protocol propagates to the next layer (its mlast / rplast source).
        # When this protocol was asked for an unpermuted output (p == 1, the
        # last non-local layer) or the shape does not tile by the stride, fall
        # back to a clean, unpermuted tensor.
        # pool_res carries the persisted values of the pooling sub-protocol
        # (mode=online): its additive share s and full permutation p; m is
        # derived here from this protocol's restored m and p_block is already
        # restored in self.p through the layer's setup kwargs.
        pool_res = pool_res or {}
        block = torch.ones(stride_shape, device=self.device)
        mp = torch.kron(self.m, block) # kronecker product
        sp = pool_res.get('s')
        sp = torch.rand_like(mp) if sp is None else sp.to(self.device)
        C, H, W = self.ishape[-3:]
        identity = isinstance(self.p, int) and self.p == 1
        tiles = H % stride_shape[0] == 0 and W % stride_shape[1] == 0
        if not identity and tiles:
            p_full = pool_res.get('p')
            if p_full is None:
                p_full, p_block = block_shuffle_perm(self.ishape, stride_shape, device=self.device)
                self.p = p_block
            else:
                p_full = p_full.to(self.device)
        else:
            p_full = 1
            self.p = 1
        pro = ProtocolServer(self.socket, self.stat, self.he, self.device)
        pro.setup(self.ishape, mp.shape, last=self.last, s=sp, m=mp, p=p_full)
        return pro
        
    def deshuflle_input(self, data: torch.Tensor) -> torch.Tensor:
        if isinstance(self.rplast, int):
            res = data
        else:
            res = deshuffle_tensor(data, self.rplast).reshape(self.ishape)
        return res

    def shuffle_output(self, data: torch.Tensor) -> torch.Tensor:
        if isinstance(self.p, int):
            res = data
        else:
            res = shuffle_tensor(data, self.p).reshape(self.oshape)
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
