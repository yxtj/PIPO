from socket import socket
# from typing import Union
# import numpy as np
# import torch

from .stat import Stat

from src.heutil import HEEngine


class LayerCommon():
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he: HEEngine, device: str) -> None:
        self.socket = socket
        self.ishape = ishape
        self.oshape = oshape
        self.he = he
        self.device = device
        self.stat = Stat()
    