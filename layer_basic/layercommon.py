from socket import socket
# from typing import Union
# import numpy as np
# import torch
from Pyfhel import Pyfhel

from .stat import Stat


class LayerCommon():
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        self.socket = socket
        self.ishape = ishape
        self.oshape = oshape
        self.he = he
        self.stat = Stat()
    