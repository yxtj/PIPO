from socket import socket
# from typing import Union
# import numpy as np
# import torch

from .stat import Stat

try:  # Pyfhel is optional; only used as a type hint here
    from Pyfhel import Pyfhel
except ImportError:  # pragma: no cover
    Pyfhel = None


class LayerCommon():
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he: Pyfhel, device: str) -> None:
        self.socket = socket
        self.ishape = ishape
        self.oshape = oshape
        self.he = he
        self.device = device
        self.stat = Stat()
    