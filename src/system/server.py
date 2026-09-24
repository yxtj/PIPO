import socket
import torch
import torch.nn as nn

from src.system import util

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
        # for shortcut layer: {fm idx: [shortcut layer idx]}
        self.dependency = {}
        for j, fs in self.shortcuts.items():
            for f in fs:
                if f in self.dependency:
                    self.dependency[f].append(j)
                else:
                    self.dependency[f] = [j]
        # self.to_buffer = [v for k,v in self.shortcuts.items()]
    
    def offline(self):
        last_non_local = util.find_last_non_local_layer(len(self.layers), self.locals)
        last_lyr = None
        data = None
        for i, lyr in enumerate(self.layers):
            name = lyr.__class__.__name__
            print('  offline {}: {}(inshape={}, outshape={}) ...'.format(i, name, lyr.ishape, lyr.oshape))
            # setup
            m = 1.0 if i == last_non_local else None
            lyr.setup(last_lyr, m)
            last_lyr = lyr
            # offline
            data = lyr.offline() # get the input of this layer (i-th intermediate result)
            if i in self.dependency:
                for j in self.dependency[i]:
                    if j != i: # the own input fm is passed directly to the layer itself
                        self.layers[j].update_offline(i, data)

    def online(self):
        for i, lyr in enumerate(self.layers):
            data = lyr.online() # get the input of this layer (i-th intermediate result)
            if i in self.dependency:
                for j in self.dependency[i]:
                    if j != i:
                        self.layers[j].update_online(i, data)
        