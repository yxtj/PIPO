from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class ConvClient(LayerClient):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)

    
class ConvServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.Conv2d)
        super().__init__(socket, ishape, oshape, layer)
    
    def offline(self) -> np.ndarray:
        t = time.time()
        rm = self.protocol.recv_offline() # recv: r'_i = r_i / m_{i-1}
        data = self.run_layer_offline(rm) # W_i * r'_i
        self.protocol.send_offline(data) # send: (W_i * r'_i) .* m_i - s_i
        self.stat.time_offline += time.time() - t
        return rm
    
    def online(self) -> torch.Tensor:
        t = time.time()
        xmr_i = self.protocol.recv_online() # recv: xmr_i = x_i - r_i / m_{i-1}
        data = self.layer(xmr_i) # W_i * (x_i - r_i / m_{i-1})
        self.protocol.send_online(data) # send: (W_i * (x_i - r_i / m_{i-1})) .* m_i + s_i
        self.stat.time_online += time.time() - t
        return xmr_i

