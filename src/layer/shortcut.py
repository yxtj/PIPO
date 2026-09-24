from socket import socket
from typing import Union
import time
import numpy as np
import torch

from src.model.dag_model import JumpOp, AddOp, ConcatOp
from .base import LayerClient, LayerServer

try:  # Pyfhel is optional; only needed when use_he is enabled
    from Pyfhel import Pyfhel
except ImportError:  # pragma: no cover
    Pyfhel = None


# abstract shortcut layer

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he, device: str,
                 srcs: list = None, own: int = None) -> None:
        super().__init__(socket, ishape, oshape, he, device)


class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str,
                 srcs: list = None, own: int = None) -> None:
        assert isinstance(layer, (JumpOp, AddOp, ConcatOp))
        super().__init__(socket, ishape, oshape, layer, device)
        self.srcs = list(srcs) # source feature-map indices (absolute, incl. the own input fm)
        self.own = own # this layer's input fm index (== flat layer index)
        self.buff = {} # used and cleaned by offline/online only

    def update_offline(self, idx:int, buff: Union[np.ndarray, torch.Tensor]) -> None:
        t0 = time.time()
        if self.buff is None:
            self.buff = {}
        self.buff[idx] = buff
        self.stat.time_offline_comp += time.time() - t0

    def update_online(self, idx:int, buff: torch.Tensor) -> None:
        t0 = time.time()
        if self.buff is None:
            self.buff = {}
        self.buff[idx] = buff
        self.stat.time_online_comp += time.time() - t0

    def online(self) -> torch.Tensor:
        t0 = time.time()
        xrm_i = self.protocol.recv_online()
        t1 = time.time()
        vals = [xrm_i if s == self.own else self.buff[s] for s in self.srcs]
        data = self.layer(*vals) if len(vals) > 1 else self.layer(vals[0])
        t2 = time.time()
        self.protocol.send_online(data)
        t3 = time.time()
        self.stat.time_online_comp += t2 - t1
        self.stat.time_online += t3 - t0
        return xrm_i


# jump layer

class JumpClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he, device: str, srcs=None, own=None) -> None:
        assert len(srcs) == 1
        super().__init__(socket, ishape, oshape, he, device, srcs, own)


class JumpServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str, srcs=None, own=None) -> None:
        assert isinstance(layer, JumpOp)
        assert len(srcs) == 1
        super().__init__(socket, ishape, oshape, layer, device, srcs, own)

    def _forward(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return data

    def offline(self) -> np.ndarray:
        t0 = time.time()
        rm_i = self.protocol.recv_offline()
        # the offline share of a jump is the source layer's offline data
        data = self.buff[self.srcs[0]]
        self.protocol.send_offline(data)
        self.buff = None
        t2 = time.time()
        self.stat.time_offline += t2 - t0
        return rm_i


# addition layer

class AdditionClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he, device: str, srcs=None, own=None) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, he, device, srcs, own)


class AdditionServer(ShortCutServer):
    def __init__(self, socket, ishape, oshape, layer, device, srcs=None, own=None) -> None:
        assert isinstance(layer, AddOp)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer, device, srcs, own)

    def _forward(self, rm_i) -> torch.Tensor:
        # rm_i (the own input share) + the buffered shares of the other sources
        if isinstance(rm_i, torch.Tensor):
            vals = [torch.as_tensor(self.buff[s]) for s in self.srcs if s != self.own]
            vals.append(rm_i)
            data = torch.stack(vals).sum(dim=0)
        else: # numpy
            vals = [self.buff[s] for s in self.srcs if s != self.own]
            vals.append(rm_i)
            data = np.stack(vals).sum(axis=0)
        return data

    def offline(self) -> np.ndarray:
        t0 = time.time()
        rm_i = self.protocol.recv_offline() # r_i/m_{i-1}
        t1 = time.time()
        data = self._forward(rm_i)
        t2 = time.time()
        self.protocol.send_offline(data)
        self.buff = None
        t3 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline += t3 - t0
        return rm_i


# concatenation layer

class ConcatenationClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he, device: str, srcs=None, own=None) -> None:
        super().__init__(socket, ishape, oshape, he, device, srcs, own)


class ConcatenationServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str, srcs=None, own=None) -> None:
        assert isinstance(layer, ConcatOp)
        super().__init__(socket, ishape, oshape, layer, device, srcs, own)
        self.dim = layer.dim

    def _forward(self, rm_i) -> torch.Tensor:
        # concatenate in the source order, rm_i sits at the own fm position
        vals = [rm_i if s == self.own else self.buff[s] for s in self.srcs]
        if isinstance(rm_i, torch.Tensor):
            data = torch.cat(vals, dim=self.dim)
        else: # numpy
            data = np.concatenate(tuple(vals), axis=self.dim)
        return data

    def offline(self) -> np.ndarray:
        t0 = time.time()
        rm_i = self.protocol.recv_offline()
        t1 = time.time()
        data = self._forward(rm_i)
        t2 = time.time()
        self.protocol.send_offline(data)
        self.buff = None
        t3 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline += t3 - t0
        return rm_i
