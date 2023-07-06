from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from torch_extension.shortcut import ShortCut
from Pyfhel import Pyfhel

class ShortCutClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, he)


class ShortCutServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, ShortCut)
        assert ishape == oshape
        super().__init__(socket, ishape, oshape, layer)
        self.other_offset = layer.otherlayer
    
    def offline(self, rm_j) -> torch.Tensor:
        t = time.time()
        rm_i = self.protocol.recv_offline() # r_i/m_{i-1}
        data = rm_i + rm_j # r_i / m_{i-1} + r_j / m_{j-1}
        self.protocol.send_offline(data)
        self.stat.time_offline += time.time() - t
        return rm_i
    
    def online(self, xmr_j) -> torch.Tensor:
        t = time.time()
        xrm_i = self.protocol.recv_online() # x_i - r_i / m_{i-1}
        data = xrm_i + xmr_j # (x_i + x_j) - (r_i / m_{i-1} - r_j / m_{j-1})
        self.protocol.send_online(data)
        self.stat.time_online += time.time() - t
        return xrm_i
    