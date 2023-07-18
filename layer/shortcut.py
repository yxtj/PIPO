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
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)


class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, ShortCut)
        super().__init__(socket, ishape, oshape, layer)
        self.other_offset = layer.relOther
        self.buff = None # used and cleaned by offline only

    def update_offline(self, buff: Union[np.ndarray, torch.Tensor]) -> None:
        self.buff = buff

    def update_online(self, buff: torch.Tensor) -> None:
        self.layer.update(buff)

    def online(self) -> torch.Tensor:
        t = time.time()
        xrm_i = self.protocol.recv_online()
        data = self.layer(xrm_i)
        # data = self.layer.forward2(self.y, xrm_i)
        self.protocol.send_online(data)
        self.stat.time_online += time.time() - t
        return xrm_i

# jump layer 

class JumpClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)


class JumpServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, Jump)
        super().__init__(socket, ishape, oshape, layer)

    def offline(self) -> torch.Tensor:
        t = time.time()
        rm_i = self.protocol.recv_offline()
        data = self.buff
        self.protocol.send_offline(data)
        self.buff = None
        self.stat.time_offline += time.time() - t
        return rm_i

# addition layer

class AdditionClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, he)


class AdditionServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, Addition)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer)

    def offline(self) -> torch.Tensor:
        t = time.time()
        rm_i = self.protocol.recv_offline() # r_i/m_{i-1}
        data = self.buff + rm_i # r_i / m_{i-1} + r_j / m_{j-1}
        self.protocol.send_offline(data)
        self.buff = None
        self.stat.time_offline += time.time() - t
        return rm_i
    
# concatenation layer

class ConcatenationClient(ShortCutClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)


class ConcatenationServer(ShortCutServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, Concatenation)
        super().__init__(socket, ishape, oshape, layer)
        self.dim = layer.dim
        self.other_offset = layer.relOther
    
    def offline(self) -> torch.Tensor:
        t = time.time()
        rm_i = self.protocol.recv_offline()
        if isinstance(rm_i, torch.Tensor):
            data = torch.cat((self.buff, rm_i), dim=self.dim)
        else: # numpy
            data = np.concatenate((self.buff, rm_i), axis=self.dim)
        self.protocol.send_offline(data)
        self.buff = None
        self.stat.time_offline += time.time() - t
        return rm_i
    