import socket
import torch
import torch.nn as nn
from Pyfhel import Pyfhel

from system import util

class Client():
    def __init__(self, socket: socket.socket, model: nn.Module, inshape: tuple, he: Pyfhel):
        self.socket = socket
        # model
        self.model = model
        self.inshape = inshape
        # he
        self.he = he
        # layers
        self.layers, self.linears, self.shortcuts, self.locals = util.make_client_model(socket, model, inshape, he)
        print("Model loaded {} layers: {} linear layers, {} local layers, {} shortcut layers.".format(
            len(self.layers), len(self.linears), len(self.locals), len(self.shortcuts)))
        # for shortcut layer
        self.to_buffer = [v for k,v in self.shortcuts.items()]
        
    def offline(self):
        for i, lyr in enumerate(self.layers):
            print('  offline {}: {} ...'.format(i, lyr.__class__.__name__))
            # setup
            # is_input_layer = i == 0
            # is_output_layer = i == len(self.layers) - 1
            lyr.setup()
            # lyr.setup(is_input_layer=is_input_layer, is_output_layer=is_output_layer)
            # offline
            lyr.offline()
            
    def online(self, data: torch.Tensor) -> torch.Tensor:
        for lyr in self.layers:
            data = lyr.online(data)
        return data

