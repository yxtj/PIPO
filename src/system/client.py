import struct

from src.comm.model_info import recv_model_info
from src.common.config import get_config, he_engine_args
from src.heutil import create_he
from src.system import util


class Client():
    def __init__(self, socket, model=None, inshape=None, he=None):
        # Constructors stay side-effect free: no recv, no layer construction.
        # setup() performs the model-info exchange and builds layers.
        self.socket = socket
        self.model = model
        self._inshape = inshape
        # a None engine (e.g. tests) falls back to the config-selected backend
        self.he = he if he is not None else create_he(*he_engine_args(get_config()))
        self.model_name = None
        self.info = None
        self.inshape = None
        self.resource = None # resource set announced by the server: save (offline) or load (online)
        self.layers = []
        self.linears = []
        self.shortcuts = {}
        self.locals = []

    def setup(self):
        """Receive the server-parsed model info and build client layers."""
        message = recv_model_info(self.socket)
        self.model_name = message.model_name
        self.info = message.info
        self.inshape = self.info.inshape
        self.resource = message.resource
        if self._inshape is not None and tuple(self._inshape) != tuple(self.inshape):
            raise ValueError(
                "client inshape {} does not match server model info {}".format(
                    tuple(self._inshape), tuple(self.inshape)))
        device = get_config().common.device
        # send the HE context once, right after the model info: the server
        # reads it before building its layers (empty for the NoHE backend)
        b_ctx = self.he.to_bytes_context()
        self.socket.sendall(struct.pack('!i', len(b_ctx)) + b_ctx)
        self.layers, self.linears, self.shortcuts, self.locals = \
            util.make_client_model_from_info(self.socket, self.info, self.he, device)
        print("Model info received: {} ({} layers, input {})".format(
            self.model_name, len(self.layers), tuple(self.inshape)))

    def setup_layers(self, res: list = None):
        """Per-layer protocol setup. With res (the layer entries of a loaded
        resource set, run mode 'online') the persisted masks replace fresh
        randomness through setup kwargs."""
        if self.info is None:
            raise RuntimeError("Client.setup() must be called before setup_layers()")
        for i, lyr in enumerate(self.layers):
            print('  setup {}: {} ...'.format(i, lyr.__class__.__name__))
            if res is not None:
                lyr.setup_res(res[i])
            else:
                lyr.setup()

    def offline(self):
        if self.info is None:
            raise RuntimeError("Client.setup() must be called before offline()")
        self.setup_layers()
        for i, lyr in enumerate(self.layers):
            print('  offline {}: {} ...'.format(i, lyr.__class__.__name__))
            lyr.offline()

    def online(self, data):
        if self.info is None:
            raise RuntimeError("Client.setup() must be called before online()")
        for lyr in self.layers:
            data = lyr.online(data)
        return data