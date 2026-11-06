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
        self.rplast = None
        self.s = None
        self.p = None # permutation rule
        self.rp = None # reverse permutation rule
        self.offline_buffer = None # used to generate random data for offline phase
    
    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None, **kwargs) -> None:
        '''
        Pure shuffle: only additive masks are used -- r on the client side
        (protecting x) and s on the server side (protecting W); there is no
        multiplicative mask. A local layer applies no new shuffle but
        propagates the pending permutation; the last non-local layer uses the
        identity permutation. s/p may be supplied through kwargs by internal
        sub-protocols (e.g. gen_mpooling).
        '''
        super().setup(ishape, oshape, last, **kwargs)
        assert last is None or isinstance(last, ProtocolServer)
        if self.is_local_layer:
            s = 0
            p = last.p if last is not None else 1
        else:
            s = kwargs.get('s')
            p = kwargs.get('p')
            if p is None and self.is_last_layer:
                p = 1
        assert p is None or isinstance(p, int) or p.numel() == int(np.prod(ishape))
        if last is None or isinstance(last.p, int):
            # no pending permutation to invert: first layer, or the previous
            # layer applies none (local layers / pooling propagate p = 1)
            self.rplast = 1
        else:
            self.rplast = gen_reverse_shuffle(last.p)
        self.s = self._gen_add_share_(s, oshape)
        if p is None:
            self.p = gen_shuffle(self.oshape, device=self.device) # shuffle matrix
        else:
            # honor the given permutation: int 1 = no shuffle (local layers,
            # pooling), or a propagated pending permutation tensor
            self.p = p
        # self.rp = torch.argsort(self.p.ravel()).reshape(self.ishape) # unshuffle matrix

    def gen_mpooling(self, stride_shape: tuple) -> ProBaseServer:
        # Block-wise shuffle is the default for pooling: the previous layer's
        # full permutation is undone first (protocol_pool.rplast), then a
        # block-wise one is applied -- the stride-sized block grid is permuted
        # as a whole and every block is shuffled per channel, so the client can
        # pool correctly while never seeing the real spatial layout. The
        # induced block permutation p_block is propagated so the next layer can
        # invert it. When this protocol was asked for an unpermuted output
        # (p == 1, the last non-local layer) or the shape does not tile by the
        # stride, fall back to a clean, unpermuted tensor.
        block = torch.ones(stride_shape, device=self.device)
        # pooling output shape: each spatial dim scaled by the stride (the
        # ones tensor only carries the shape)
        pool_oshape = torch.kron(torch.ones(self.oshape, device=self.device), block).shape
        C, H, W = self.ishape[-3:]
        identity = isinstance(self.p, int) and self.p == 1
        tiles = H % stride_shape[0] == 0 and W % stride_shape[1] == 0
        if not identity and tiles:
            p_full, p_block = block_shuffle_perm(self.ishape, stride_shape, device=self.device)
            self.p = p_block
        else:
            p_full = 1
            self.p = 1
        pro = ProtocolServer(self.socket, self.stat, self.he, self.device)
        pro.setup(self.ishape, pool_oshape, last=self.last, p=p_full)
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
        t2 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline_recv += t1 - t0
        return data
        
    def send_offline(self, data: np.ndarray) -> None:
        t0 = time.time()
        data = self.shuffle_output(data)
        data += self.s
        t1 = time.time()
        self.basic_send_offline(data)
        t2 = time.time()
        self.stat.time_offline_comp += t1 - t0
        self.stat.time_offline_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        '''
        Receive p_last(x - r) from client; return the unshuffled x - r.
        '''
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        data = self.deshuflle_input(data)
        t2 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        self.stat.time_online_comp += t2 - t1
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        '''
        Input data = W(x - r); send p(data) - s to client.
        '''
        t0 = time.time()
        data = self.shuffle_output(data)
        data -= self.s
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1
