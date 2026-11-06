from .base import LocalLayerClient, LocalLayerServer

from socket import socket
import time
import torch
from src.heutil import HEEngine
class FlattenClient(LocalLayerClient):
    ltype = 'flatten'

    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he: HEEngine, lyr: torch.nn.Module, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
        self.layer = lyr
    
    def online(self, xm) -> torch.Tensor:
        t0 = time.time()
        data = self.layer(xm)
        t1 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online += t1 - t0
        return data


class FlattenServer(LocalLayerServer):
    ltype = 'flatten'

    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.Flatten)
        super().__init__(socket, ishape, oshape, layer, device)