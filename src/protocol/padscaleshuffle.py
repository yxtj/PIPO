'''Pad + scale + shuffle protocol: the masked output is first inflated with k
random elements resampled from the data itself, then every element (real and
pad alike) is scaled by a multiplicative mask and permuted. The client cannot
distinguish real elements from pads, cannot recover the true coordinates, and
only ever holds the product ``Wx * m`` with an unknown ``m``. It combines
padshuffle's inflation with scaleshuffle's multiplicative blinding.'''

from .ptobase import ProBaseServer, ProBaseClient, NumberType
from .crypto import block_shuffle_perm, gen_shuffle, gen_reverse_shuffle, shuffle_tensor, deshuffle_tensor, pad_tensor, depad_tensor, pad_count, gen_mul_share
from typing import Union

import socket

import time
import torch
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
        self.kin = 0 # pad count embedded in the tensors this client holds

    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        super().setup(ishape, oshape, **kwargs)
        # no handshake: the client derives the pad count of the previous
        # layer's output from pad_ratio, exactly as the server computes it
        self.kin = pad_count(int(np.prod(ishape)), kwargs['pad_ratio'])
        n = int(np.prod(ishape))
        self.r = self._gen_add_share_(kwargs.get('r'), (1, n + self.kin))

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
        # flatten to the wire shape (1, numel + kin): the client holds either
        # the raw input or the previous layer's inflated, scaled, permuted
        # output. The derived kin can only over-estimate (e.g. a maxpool hands
        # over a clean tensor), so clip r to the actual tensor; the server
        # trims the same excess from the offline share.
        data = data.reshape(1, -1)
        data = data - self.r[:, :data.numel()]
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
    res_fields = ('s', 'm', 'p', 'k', 'mp_pad')

    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.mlast = None
        self.rplast = None
        self.s = None
        self.m = None # clean multiplicative mask handed to the next layer
        self.mp = None # m expanded over data + pads (None when p is identity)
        self.mp_pad = None # the pads' independent random mask (mp = cat(m, mp_pad))
        self.p = None # permutation rule (int = identity)
        self.k = 0 # pad count inserted into this layer's output

    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None, **kwargs) -> None:
        '''
        Pad + scale + shuffle: the masked output is inflated with k random
        elements resampled from the data itself, then every element is scaled
        by a multiplicative mask and permuted by p. Real positions use the
        layer's clean mask m (which is propagated to the next layer for
        ``/mlast``); pad positions use an independent random mask, so the
        client cannot tell them apart.
        A local layer applies no new shuffle or padding but propagates the
        pending permutation, pad count and mask; the last non-local layer uses
        the identity permutation and no pads. Otherwise p is random and k is
        derived from pad_ratio, unless supplied through kwargs by an internal
        sub-protocol.
        setup performs no socket communication: the client derives the
        previous layer's pad count from pad_ratio on its side.
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
            k = kwargs.get('k')
        else:
            s = kwargs.get('s')
            m = kwargs.get('m')
            p = kwargs.get('p')
            k = kwargs.get('k')
            if m is None and self.is_last_layer:
                m = 1.0
            if p is None and self.is_last_layer:
                p = 1
        assert p is None or isinstance(p, int) or torch.is_tensor(p)
        if last is None or isinstance(last.p, int):
            # no pending permutation to invert: first layer, or the previous
            # layer applies none (local layers / pooling propagate p = 1)
            self.rplast = 1
        else:
            # 1-D inverse permutation over last's inflated output
            self.rplast = gen_reverse_shuffle(last.p)
        self.s = self._gen_add_share_(s, oshape)
        self.m = self._gen_mul_share_(m, oshape)
        n = int(np.prod(oshape))
        if p is None:
            ratio = kwargs['pad_ratio']
            self.k = pad_count(n, ratio)
            self.p = gen_shuffle((1, n + self.k), device=self.device) # permutes data + pads
        elif isinstance(p, int):
            assert k is None or k == 0, "identity permutation carries no pads"
            self.p = p
            self.k = 0
        else:
            # honor the given permutation: propagated by local layers (k
            # defaults to last's pad count) or given explicitly with its own
            # pad count (pooling sub-protocol)
            self.p = p
            self.k = k if k is not None else (last.k if last is not None else 0)
        assert isinstance(self.p, int) or self.p.numel() == n + self.k, \
            "permutation must cover numel(oshape) data elements plus k pads"
        if isinstance(self.p, int) or self.is_local_layer:
            self.mp = None
            self.mp_pad = None
        else:
            # real positions keep m, pads get an independent mask of the same
            # distribution so the whole inflated tensor is uniformly scaled.
            # mp_pad is an offline random value: it is persisted in the
            # resource files (mode=online restores it through setup kwargs)
            # so that repeated runs on the same resource stay identical.
            pad_m = kwargs.get('mp_pad')
            pad_m = pad_m.to(self.device) if pad_m is not None else None
            if pad_m is None and self.k:
                pad_m = gen_mul_share((1, self.k), device=self.device)
            self.mp_pad = pad_m
            flat_m = self.m.reshape(1, -1)
            self.mp = flat_m if pad_m is None else torch.cat([flat_m, pad_m], dim=1)

    def gen_mpooling(self, stride_shape: tuple, pool_res: dict = None) -> ProBaseServer:
        # Block-wise shuffle (same policy as the shuffle/padshuffle protocols):
        # the previous layer's full permutation is undone first (deshuffle +
        # depad in protocol_pool.recv_*), then a block-wise one is applied to
        # the masked tensor the client pools on. Because mp = kron(m, ones) is
        # constant inside each pooling window, maxpool(p_full(x*m)) ==
        # p_block(maxpool(x)*m): the client-held pooled tensor carries exactly
        # the pending permutation p_block and the multiplicative mask m this
        # protocol propagates to the next layer. The pooling sub-protocol emits
        # a pure block permutation (k = 0: no inflation), so this protocol
        # propagates k = 0 for the client-held pooled tensor.
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
        self.k = 0 # the pooled output carries no pads, only the permutation
        self.mp_pad = None
        pro = ProtocolServer(self.socket, self.stat, self.he, self.device)
        pro.setup(self.ishape, mp.shape, last=self.last, s=sp, m=mp, p=p_full, k=0)
        return pro

    def deshuflle_input(self, data: torch.Tensor) -> torch.Tensor:
        '''Undo the previous layer's permutation, drop its pads, then remove
        its multiplicative mask. The client's derived pad count can
        over-estimate after a maxpool (which hands over a clean tensor), so
        any excess in the offline share is trimmed before the reverse
        permutation is applied.'''
        n = int(np.prod(self.ishape))
        if isinstance(self.rplast, int):
            # no pending permutation and no pads: keep the real prefix
            data = data.reshape(1, -1)[:, :n].reshape(self.ishape)
        else:
            # flatten first: the reverse pattern may be n-D (a pooled block
            # permutation, e.g. (1, C, h, w)) while depad works on the 1-D
            # concatenation layout (real elements first, pads last)
            data = data.reshape(1, -1)[:, :self.rplast.numel()]
            res = deshuffle_tensor(data, self.rplast).reshape(1, -1)
            # rplast restores the concatenation order (real elements first,
            # pads last): keep the real prefix
            data = depad_tensor(res, (1, n)).reshape(self.ishape)
        if not (isinstance(self.mlast, (int, float)) and self.mlast == 1):
            data /= self.mlast
        return data

    def pad_scale_shuffle_output(self, data: torch.Tensor) -> torch.Tensor:
        '''Inflate the masked data with k data-dependent pads, scale the whole
        inflated tensor by mp (real positions m, pads an independent mask of
        the same distribution), then permute.'''
        if isinstance(self.p, int):
            # no permutation and no pads, but the multiplicative mask still
            # applies (the last layer uses m = 1)
            return data * self.m
        n = data.numel()
        padded = pad_tensor(data.reshape(1, -1), (1, n + self.k), device=self.device)
        padded = padded * self.mp
        padded = shuffle_tensor(padded, self.p)
        # without pads the inflated tensor is exactly the clean shape, so keep
        # it n-D: a pooling client must be able to run maxpool on it
        if self.k == 0:
            return padded.reshape(self.oshape)
        return padded

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
        data = data + self.s # additive mask in the clean coordinate system
        data = self.pad_scale_shuffle_output(data)
        t1 = time.time()
        self.basic_send_offline(data)
        t2 = time.time()
        self.stat.time_offline_comp += t1 - t0
        self.stat.time_offline_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        '''
        Receive the previous layer's inflated p_last(x*m_last - r); return the
        unshuffled, de-padded, de-scaled x - r/mlast.
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
        Input data = W(x - r/mlast); send p((concat(data - s, pads)) * mp) to
        the client.
        '''
        t0 = time.time()
        data = data - self.s # additive mask in the clean coordinate system
        data = self.pad_scale_shuffle_output(data)
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1