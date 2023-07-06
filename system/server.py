import socket
import torch
import torch.nn as nn

from system import util

class Server():
    def __init__(self, socket: socket.socket, model: nn.Module, inshape: tuple):
        self.socket = socket
        # model
        self.model = model
        self.inshape = inshape
        # layers
        self.layers, self.linears, self.shortcuts, self.locals = util.make_server_model(socket, model, inshape)
        print("Model loaded {} layers: {} linear layers, {} local layers, {} shortcut layers.".format(
            len(self.layers), len(self.linears), len(self.locals), len(self.shortcuts)))
        # for shortcut layer
        self.to_buffer = [v for k,v in self.shortcuts.items()]
    
    def offline(self):
        last_non_local = util.find_last_non_local_layer(len(self.layers), self.locals)
        buffer = {}
        last_lyr = None
        for i, lyr in enumerate(self.layers):
            name = lyr.__class__.__name__
            print('  offline {}: {}(inshape={}, outshape={}) ...'.format(i, name, lyr.ishape, lyr.oshape))
            # setup
            m = 1.0 if i == last_non_local else None
            lyr.setup(last_lyr, m)
            last_lyr = lyr
            # print("  m :", mlast)
            # offline
            if i in self.shortcuts:
                idx = self.shortcuts[i]
                t = lyr.offline(buffer[idx])
            else:
                t = lyr.offline()
            # buffer
            if i in self.to_buffer:
                buffer[i] = t
            
    def online(self):
        buffer = {}
        for i, lyr in enumerate(self.layers):
            if i in self.shortcuts:
                idx = self.shortcuts[i]
                t = lyr.online(buffer[idx])
            else:
                t = lyr.online()
            if i in self.to_buffer:
                buffer[i] = t
        