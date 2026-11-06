import socket
import struct
import torch
import torch.nn as nn

from src.comm.model_info import send_model_info
from src.common.config import get_config
from src.common.model_info import build_model_info
from src.heutil import create_he
from src.system import util


class Server():
    def __init__(self, socket: socket.socket, model: nn.Module, inshape: tuple,
                 model_name: str = None):
        # Constructors stay side-effect free: no I/O, no shape analysis, no
        # layer construction. setup() does all of that after construction.
        self.socket = socket
        self.model = model
        self.inshape = inshape
        self.model_name = model_name
        self.info = None
        self.he = None
        self.layers = []
        self.linears = []
        self.shortcuts = {}
        self.locals = []
        self.dependency = {}

    def setup(self):
        """Parse the model, send its info to the client, and build server layers."""
        if self.model_name is None:
            self.model_name = get_config().common.model.name
        self.info = build_model_info(self.model, self.inshape)
        send_model_info(self.socket, self.model_name, self.info)
        print("Model info sent: {} ({} layers, input {})".format(
            self.model_name, len(self.info.steps), tuple(self.inshape)))
        # one shared engine for all layers; the client sends its HE context
        # once, right after receiving the model info (empty for NoHE)
        self.he = create_he()
        ctx_len = struct.unpack('!i', self.socket.recv(4))[0]
        b_ctx = self.socket.recv(ctx_len)
        self.he.from_bytes_context(b_ctx)
        self.layers, self.linears, self.shortcuts, self.locals = \
            util.make_server_model(self.socket, self.model, self.inshape, self.he)
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

    def offline(self):
        if self.info is None:
            raise RuntimeError("Server.setup() must be called before offline()")
        last_non_local = util.find_last_non_local_layer(len(self.layers), self.locals)
        last_lyr = None
        data = None
        for i, lyr in enumerate(self.layers):
            name = lyr.__class__.__name__
            print('  offline {}: {}(inshape={}, outshape={}) ...'.format(i, name, lyr.ishape, lyr.oshape))
            # setup
            # the last non-local layer hands its plaintext result to the
            # client, so the protocol must not mask or permute its output
            lyr.setup(last_lyr, is_last_layer=(i == last_non_local))
            last_lyr = lyr
            # offline
            data = lyr.offline() # get the input of this layer (i-th intermediate result)
            # handle shortcut connections: update the input buffer of shortcut-end layer
            if i in self.dependency:
                for j in self.dependency[i]:
                    if j != i: # the own input fm is passed directly to the layer itself
                        self.layers[j].update_offline(i, data)

    def online(self):
        if self.info is None:
            raise RuntimeError("Server.setup() must be called before online()")
        for i, lyr in enumerate(self.layers):
            data = lyr.online() # get the input of this layer (i-th intermediate result)
            if i in self.dependency:
                for j in self.dependency[i]:
                    if j != i:
                        self.layers[j].update_online(i, data)