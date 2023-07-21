from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from torch_extension.shortcut import ShortCut, Jump, Addition, Concatenation
from Pyfhel import Pyfhel

# abstract shortcut layer

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)


class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, ShortCut)
        super().__init__(socket, ishape, oshape, layer, device)
        self.other_offset = layer.relOther
        self.buff = {} # used and cleaned by offline only

    def update_offline(self, ridx:int, buff: Union[np.ndarray, torch.Tensor]) -> None:
        self.buff[ridx] = buff

    def update_online(self, ridx:int, buff: torch.Tensor) -> None:
        self.layer.update(ridx, buff)

    def online(self) -> torch.Tensor:
        t = time.time()
        xrm_i = self.protocol.recv_online()
        data = self.layer(xrm_i)
        self.protocol.send_online(data)
        self.stat.time_online += time.time() - t
        return xrm_i

# jump layer 

class JumpClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)


class JumpServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, Jump)
        super().__init__(socket, ishape, oshape, layer, device)

    def offline(self) -> np.ndarray:
        t = time.time()
        rm_i = self.protocol.recv_offline()
        data = self.buff[self.other_offset[0]]
        self.protocol.send_offline(data)
        self.buff = None
        self.stat.time_offline += time.time() - t
        return rm_i

# addition layer

class AdditionClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, he, device)


class AdditionServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, Addition)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer, device)

    def offline(self) -> np.ndarray:
        t = time.time()
        rm_i = self.protocol.recv_offline() # r_i/m_{i-1}
        # data = self.buff + rm_i # r_i / m_{i-1} + r_j / m_{j-1}
        self.buff[-1] = rm_i
        if isinstance(rm_i, torch.Tensor):
            data = torch.stack(tuple(self.buff.values())).sum(dim=0)
        else: # numpy
            data = np.stack(tuple(self.buff.values())).sum(axis=0)
        self.protocol.send_offline(data)
        self.buff = None
        self.stat.time_offline += time.time() - t
        return rm_i
    
# concatenation layer

class ConcatenationClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)


class ConcatenationServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, Concatenation)
        super().__init__(socket, ishape, oshape, layer, device)
        self.dim = layer.dim
        self.other_offset = layer.relOther
        self.buff = {e:None for e in layer.order} # used and cleaned by offline only
    
    def offline(self) -> np.ndarray:
        t = time.time()
        rm_i = self.protocol.recv_offline()
        self.buff[-1] = rm_i
        if isinstance(rm_i, torch.Tensor):
            data = torch.cat(tuple(self.buff.values()), dim=self.dim)
        else: # numpy
            data = np.concatenate((tuple(self.buff.values())), axis=self.dim)
        self.protocol.send_offline(data)
        self.buff = None
        self.stat.time_offline += time.time() - t
        return rm_i
    