from .ptobase import ProBaseServer, ProBaseClient, NumberType
from typing import Union

import socket
import torch
import time
import numpy as np
import comm
from Pyfhel import Pyfhel

from layer_basic.stat import Stat

__ALL__ = ['ProtocolClient', 'ProtocolServer']

class ProtocolClient(ProBaseClient):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
    
    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        super().setup(ishape, oshape)

    def send_offline(self) -> None:
        t0 = time.time()
        data = self._gen_add_share_(None, self.ishape)
        nbyte = comm.send_torch(self.socket, data)
        t1 = time.time()
        self.stat.byte_offline_recv += nbyte
        self.stat.time_offline_send += t1 - t0
    
    def recv_offline(self) -> torch.Tensor:
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        t1 = time.time()
        self.stat.byte_offline_recv += nbyte
        self.stat.time_offline_recv += t1 - t0
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        return data


class ProtocolServer(ProBaseServer):
    def __init__(self, s: socket.socket, stat: Stat, he: Pyfhel, device: str='cpu'):
        super().__init__(s, stat, he, device)
    
    def setup(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None, **kwargs) -> None:
        super().setup(ishape, oshape, last)

    def setup_local(self, ishape: tuple, oshape: tuple, last: ProBaseServer=None, 
            ltype: str='activation',  **kwargs) -> None:
        return self.setup(ishape, oshape, last, **kwargs)

    def gen_mpooling(self, stride_shape: tuple) -> ProBaseServer:
        return self

    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        self.stat.byte_offline_recv += nbyte
        self.stat.time_offline_recv += t1 - t0
        return data
    
    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        t1 = time.time()
        nbyte = comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.byte_offline_send += nbyte
        self.stat.time_offline_send += t2 - t1
    
    def recv_online(self) -> torch.Tensor:
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        data = data.to(self.device)
        t1 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_send += t2 - t1
