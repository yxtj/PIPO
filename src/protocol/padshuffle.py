'''Pad + shuffle protocol: the masked output is inflated with k random
elements resampled from the data itself and then permuted, so the client
cannot distinguish real elements from pads.'''

from .ptobase import ProBaseServer, ProBaseClient, NumberType
from .crypto import block_shuffle_perm, gen_shuffle, gen_reverse_shuffle, shuffle_tensor, deshuffle_tensor, pad_tensor, depad_tensor, pad_count
from typing import Union

import socket

import time
import torch
import numpy as np
import src.comm as comm
try:  # Pyfhel is optional; only needed when use_he is enabled
    from Pyfhel import Pyfhel
except ImportError:  # pragma: no cover
    Pyfhel = None
from src.layer_basic.stat import Stat
from src.common.config import get_config

__ALL__ = ['ProtocolClient', 'ProtocolServer']



class ProtocolClient(ProBaseClient):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.r = None
        self.pre = None
        self.kin = 0 # pad count embedded in the tensors this client holds

    def setup(self, ishape:tuple, oshape:tuple, r: NumberType=None, **kwargs) -> None:
        super().setup(ishape, oshape)
        # no handshake: the client derives the pad count of the previous
        # layer's output from pad_ratio, exactly as the server computes it
        self.kin = pad_count(int(np.prod(ishape)), get_config().common.pad_ratio)
        n = int(np.prod(ishape))
        self.r = self._gen_add_share_(r, (1, n + self.kin))

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
        # flatten to the wire shape (1, numel + kin): what the client holds is
        # either the raw input or the previous layer's inflated output.
        # The derived kin can only over-estimate (a maxpool hands over a
        # clean tensor), so clip r to the actual tensor; the server trims the
        # same excess from the offline share.
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
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.rplast = None
        self.s = None
        self.p = None # permutation rule (int = identity)
        self.k = 0 # pad count inserted into this layer's output

    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None,
        s: NumberType=None, m: NumberType=None, p=None, k: int=None,
        **kwargs) -> None:
        '''
        Shuffle + pad: the masked output is inflated with k random elements
        resampled from the data itself, then permuted by p, so the client
        cannot distinguish real elements from pads.
        If p is None: generate p randomly; k = round(pad_ratio * numel(oshape)).
        If p is 1: no shuffle and no padding (k = 0).
        If p is a tensor: k defaults to the pad count propagated by `last`
        (local layers), unless given explicitly (the pooling sub-protocol
        passes a block permutation with k = 0: shuffle without inflation).
        A consistency check rejects any p whose size does not cover
        numel(oshape) + k data and pad elements.
        setup performs no socket communication: the client derives the
        previous layer's pad count from pad_ratio on its side.
        The ``m`` argument is kept for compatibility with the layer setup
        calls and ignored.
        '''
        assert last is None or isinstance(last, ProtocolServer)
        super().setup(ishape, oshape, last)
        assert p is None or isinstance(p, int) or torch.is_tensor(p)
        if last is None or isinstance(last.p, int):
            # no pending permutation to invert: first layer, or the previous
            # layer applies none (local layers / pooling propagate p = 1)
            self.rplast = 1
        else:
            # 1-D inverse permutation over last's inflated output
            # (numel(last.oshape) + last.k elements)
            self.rplast = gen_reverse_shuffle(last.p)
        self.s = self._gen_add_share_(s, oshape)
        if p is None:
            ratio = get_config().common.pad_ratio
            n = int(np.prod(oshape))
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
        assert isinstance(self.p, int) or self.p.numel() == int(np.prod(oshape)) + self.k, \
            "permutation must cover numel(oshape) data elements plus k pads"

    def setup_local(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None,
            ltype: str='activation',  **kwargs) -> None:
        s = 0
        # a local layer applies no new shuffle or padding, but the state of
        # the last shuffled layer persists in the client-held tensor —
        # propagate it so the next layer's rplast inverts the right
        # permutation and drops the right number of pads
        p = last.p if last is not None else 1
        self.setup(ishape, oshape, last, s, p=p)

    def gen_mpooling(self, stride_shape: tuple) -> ProBaseServer:
        # Block-wise shuffle (same policy as the shuffle protocol): the
        # previous layer's full permutation is undone first (deshuffle +
        # depad in protocol_pool.recv_*), then a block-wise one is applied to
        # the masked tensor the client pools on; pooling windows never cross a
        # block boundary, so maxpool(p_full(x - r)) == p_block(maxpool(x - r)).
        # The pooling sub-protocol emits a pure block permutation (k = 0:
        # shuffle without inflation), so the client-held pooled tensor is
        # exactly p_block(maxpool(x)) -- this protocol then propagates p_block
        # as the pending permutation for the next layer's rplast.
        # When this protocol was asked for an unpermuted output (p == 1, the
        # last non-local layer) or the shape does not tile by the stride, fall
        # back to a clean, unpermuted tensor.
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
            self.k = 0 # the pooled output carries no pads, only the permutation
        else:
            p_full = 1
            self.p = 1
            self.k = 0
        pro = ProtocolServer(self.socket, self.stat, self.he, self.device)
        pro.setup(self.ishape, pool_oshape, last=self.last, p=p_full, k=0)
        return pro

    def deshuflle_input(self, data: torch.Tensor) -> torch.Tensor:
        '''Undo the previous layer's permutation and drop its pad elements.
        The client's derived pad count can over-estimate after a maxpool
        (which hands over a clean tensor), so any excess in the offline share
        is trimmed before the reverse permutation is applied.'''
        n = int(np.prod(self.ishape))
        if isinstance(self.rplast, int):
            # no pending permutation and no pads: keep the real prefix
            return data.reshape(1, -1)[:, :n].reshape(self.ishape)
        # flatten first: the reverse pattern may be n-D (a pooled block
        # permutation, e.g. (1, C, h, w)) while depad works on the 1-D
        # concatenation layout (real elements first, pads last)
        data = data.reshape(1, -1)[:, :self.rplast.numel()]
        res = deshuffle_tensor(data, self.rplast).reshape(1, -1)
        # rplast restores the concatenation order (real elements first,
        # pads last): keep the real prefix
        res = depad_tensor(res, (1, n))
        return res.reshape(self.ishape)

    def pad_shuffle_output(self, data: torch.Tensor) -> torch.Tensor:
        '''Inflate the masked data with k data-dependent pads, then permute.'''
        if isinstance(self.p, int):
            return data
        n = data.numel()
        padded = pad_tensor(data.reshape(1, -1), (1, n + self.k), device=self.device)
        return shuffle_tensor(padded, self.p)

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
        data = self.pad_shuffle_output(data)
        t1 = time.time()
        self.basic_send_offline(data)
        t2 = time.time()
        self.stat.time_offline_comp += t1 - t0
        self.stat.time_offline_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        '''
        Receive the previous layer's inflated p_last(x - r); return the
        unshuffled, de-padded x - r.
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
        Input data = W(x - r); send p(concat(data - s, pads)) to client.
        '''
        t0 = time.time()
        data = data - self.s # additive mask in the clean coordinate system
        data = self.pad_shuffle_output(data)
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1
