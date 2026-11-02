'''Pure pad protocol: the masked output is inflated with k random elements
resampled from the data itself and appended after the real elements, without
any permutation (the client knows where the pads are). It is the no-shuffle
baseline of the padshuffle protocol.'''

from .ptobase import ProBaseServer, ProBaseClient, NumberType
from .crypto import pad_tensor, pad_count
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
        self.s = None
        self.k = 0 # pad count appended to this layer's output
        self.kin = 0 # pad count carried by the incoming tensor

    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None,
        s: NumberType=None, m: NumberType=None, k: int=None, **kwargs) -> None:
        '''
        Pure pad: the masked output is inflated with k random elements
        resampled from the data itself and appended after the real elements.
        No permutation is ever applied, so the client knows the pad positions
        (this protocol is the no-shuffle baseline of padshuffle).
        If k is None: k = round(pad_ratio * numel(oshape)) and the output is
        inflated, except for the last non-local layer: the assembler marks it
        with the compatibility flag p=1 (its plaintext result goes to the
        client) and its output carries no pads. Local layers propagate the
        pending pad count explicitly.
        A consistency check rejects any k < 0.
        The client derives the same count from pad_ratio on its side, so
        setup performs no socket communication.
        The ``m`` argument is kept for compatibility with the layer setup
        calls and ignored.
        '''
        assert last is None or isinstance(last, ProtocolServer)
        super().setup(ishape, oshape, last)
        # remember how many pads the incoming tensor carries (0 for the first
        # layer); no handshake, the client derives it from pad_ratio
        self.kin = last.k if last is not None else 0
        self.s = self._gen_add_share_(s, oshape)
        if k is not None:
            assert k >= 0, "pad count must not be negative"
            self.k = k
        elif kwargs.get('p') == 1:
            # the assembler's "hand the plaintext result to the client" marker
            self.k = 0
        else:
            self.k = pad_count(int(np.prod(oshape)), get_config().common.pad_ratio)

    def setup_local(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None,
            ltype: str='activation',  **kwargs) -> None:
        s = 0
        # a local layer appends no pad of its own, but the padding of the last
        # non-local layer persists in the client-held tensor — propagate its
        # count so the next layer reports the right kin
        k = last.k if last is not None else 0
        self.setup(ishape, oshape, last, s, k=k)

    def gen_mpooling(self, stride_shape: tuple) -> ProBaseServer:
        # Pure pad has no permutation to protect pooling, so the pooling
        # sub-protocol stays clean: the previous layer's pads are dropped by
        # this protocol's recv_*, and the client pools on an unpermuted
        # masked tensor. Its output carries no pads (k = 0), which this
        # protocol propagates as the pending pad count for the next layer.
        block = torch.ones(stride_shape, device=self.device)
        # pooling output shape: each spatial dim scaled by the stride (the
        # ones tensor only carries the shape)
        pool_oshape = torch.kron(torch.ones(self.oshape, device=self.device), block).shape
        self.k = 0
        pro = ProtocolServer(self.socket, self.stat, self.he, self.device)
        pro.setup(self.ishape, pool_oshape, last=self.last, k=0)
        return pro

    def depad_input(self, data: torch.Tensor) -> torch.Tensor:
        '''Drop the pads carried by the previous layer's output: they are
        appended after the real elements, so the real prefix is kept.
        The count is not trusted blindly: the client's derived kin can
        over-estimate after a maxpool (which hands over a clean tensor), so
        any excess is trimmed here as well.'''
        n = int(np.prod(self.ishape))
        return data.reshape(1, -1)[:, :n].reshape(self.ishape)

    def pad_output(self, data: torch.Tensor) -> torch.Tensor:
        '''Inflate the masked data with k data-dependent pads (no shuffle).'''
        if self.k == 0:
            return data.reshape(self.oshape)
        n = data.numel()
        return pad_tensor(data.reshape(1, -1), (1, n + self.k), device=self.device)

    def recv_offline(self) -> torch.Tensor:
        t0 = time.time()
        data = self.basic_recv_offline()
        t1 = time.time()
        data = self.depad_input(data)
        t2 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline_recv += t1 - t0
        return data

    def send_offline(self, data: np.ndarray) -> None:
        t0 = time.time()
        data = data + self.s # additive mask in the clean coordinate system
        data = self.pad_output(data)
        t1 = time.time()
        self.basic_send_offline(data)
        t2 = time.time()
        self.stat.time_offline_comp += t1 - t0
        self.stat.time_offline_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        '''
        Receive the previous layer's inflated x - r; return the de-padded
        x - r (in the clean coordinate system, no permutation).
        '''
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        data = self.depad_input(data)
        t2 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        self.stat.time_online_comp += t2 - t1
        return data

    def send_online(self, data: torch.Tensor) -> None:
        '''
        Input data = W(x - r); send concat(data - s, pads) to client.
        '''
        t0 = time.time()
        data = data - self.s # additive mask in the clean coordinate system
        data = self.pad_output(data)
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1