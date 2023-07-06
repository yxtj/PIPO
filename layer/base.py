from socket import socket
from typing import Union
import time
import torch
import numpy as np
from Pyfhel import Pyfhel

from layer_basic import LayerCommon
from protocol import ProtocolClient, ProtocolServer

__all__ = ['LayerClient', 'LayerServer', 'LocalLayerClient', 'LocalLayerServer']


class LayerClient(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        # self.is_input_layer = False
        # self.is_output_layer = False
        self.protocol = ProtocolClient(socket, self.stat, he)
    
    def setup(self, **kwargs):
        t = time.time()
        self.protocol.setup(self.ishape, self.oshape, r = None, **kwargs)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> None:
        t = time.time()
        # print("r", self.protocol.r)
        self.protocol.send_offline()
        # wait for the server to finish processing
        self.protocol.recv_offline()
        # print("pre", self.protocol.pre)
        self.stat.time_offline += time.time() - t
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        # print("xm", xm)
        self.protocol.send_online(xm)
        # wait for the server to finish processing
        data = self.protocol.recv_online()
        # print("wxm", data)
        self.stat.time_online += time.time() - t
        return data
    
    
class LayerServer(LayerCommon):
    def __init__(self, socket:socket, ishape:tuple, oshape:tuple, layer:torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, Pyfhel())
        self.layer = layer
        self.protocol = ProtocolServer(socket, self.stat, self.he)
    
    def setup(self, last_lyr: LayerCommon, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        t = time.time()
        assert last_lyr is None or isinstance(last_lyr, LayerServer)
        last_pto = last_lyr.protocol if last_lyr is not None else None
        self.protocol.setup(self.ishape, self.oshape, s=None, m=m, last=last_pto)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> np.ndarray:
        '''
        Return the received data from client
        '''
        raise NotImplementedError
    
    def online(self) -> torch.Tensor:
        '''
        Return the received data from client
        '''
        raise NotImplementedError

    def run_layer_offline(self, data:torch.Tensor) -> torch.Tensor:
        bias = self.layer.bias
        self.layer.bias = None
        data = self.layer(data)
        self.layer.bias = bias
        return data


# %% local layer specialization

class LocalLayerClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
    
    def offline(self) -> None:
        return

    def online(self, xm) -> torch.Tensor:
        raise NotImplementedError

class LocalLayerServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        super().__init__(socket, ishape, oshape, layer)
    
    def setup(self, last_lyr: LayerCommon, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        # forward the last layer's m
        t = time.time()
        assert isinstance(last_lyr, LayerServer)
        last_pto = last_lyr.protocol if last_lyr is not None else None
        m = last_pto.m if last_pto is not None else m
        self.protocol.setup(self.ishape, self.oshape, s=0, m=m, last=last_pto)
        self.stat.time_offline += time.time() - t
        
    def offline(self) -> None:
        return
    
    def online(self) -> None:
        return

