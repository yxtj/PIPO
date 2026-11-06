from socket import socket
import time
import torch
import numpy as np

from src.layer_basic import LayerCommon
from src.protocol import ProtocolClient, ProtocolServer

from src.heutil import HEEngine

__all__ = ['LayerClient', 'LayerServer', 'LocalLayerClient', 'LocalLayerServer']


class LayerClient(LayerCommon):
    # semantic layer kind, interpreted by the protocol module
    ltype = 'linear'

    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he: HEEngine, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
        # self.is_input_layer = False
        # self.is_output_layer = False
        self.protocol = ProtocolClient(socket, self.stat, he, device)
    
    def setup(self, **kwargs):
        t = time.time()
        self.protocol.setup(self.ishape, self.oshape, ltype=self.ltype, **kwargs)
        self.stat.time_offline += time.time() - t
    
    def offline(self) -> None:
        t0 = time.time()
        # print("r", self.protocol.r)
        self.protocol.send_offline()
        # wait for the server to finish processing
        self.protocol.recv_offline()
        t2 = time.time()
        # print("pre", self.protocol.pre)
        self.stat.time_offline += t2 - t0
    
    def online(self, xm) -> torch.Tensor:
        t0 = time.time()
        # print("xm", xm)
        self.protocol.send_online(xm)
        # wait for the server to finish processing
        data = self.protocol.recv_online()
        t2 = time.time()
        # print("wxm", data)
        self.stat.time_online += t2 - t0
        return data
    
    
class LayerServer(LayerCommon):
    # semantic layer kind, interpreted by the protocol module
    ltype = 'linear'

    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he: HEEngine, layer: torch.nn.Module, device: str) -> None:
        # he is the server-wide shared engine created in Server.setup(); the
        # client-side context is injected once before layers are built
        super().__init__(socket, ishape, oshape, he, device)
        self.layer = layer
        self.protocol = ProtocolServer(socket, self.stat, self.he, device)
    
    def setup(self, last_lyr: LayerCommon, is_last_layer: bool=False, **kwargs) -> None:
        t = time.time()
        assert last_lyr is None or isinstance(last_lyr, LayerServer)
        last_pto = last_lyr.protocol if last_lyr is not None else None
        self.protocol.setup(
            self.ishape, self.oshape,
            last=last_pto,
            ltype=self.ltype,
            is_local_layer=isinstance(self, LocalLayerServer),
            is_last_layer=is_last_layer,
            **kwargs,
        )
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
    ltype = 'activation'

    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he: HEEngine, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)

    def offline(self) -> None:
        return

    def online(self, xm) -> torch.Tensor:
        raise NotImplementedError

class LocalLayerServer(LayerServer):
    ltype = 'activation'

    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he: HEEngine, layer: torch.nn.Module, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, layer, device)
        
    def offline(self) -> None:
        return
    
    def online(self) -> None:
        return

