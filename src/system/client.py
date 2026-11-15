import struct

from src.comm.model_info import recv_model_info
from src.comm.resource import recv_resource_ids, send_resource_id
from src.common.config import he_engine_args
from src.heutil import create_he
from src.system import util
from src.system.resource import choose_index, load_resource_set_client


class Client():
    def __init__(self, socket, cfg, model=None, inshape=None, he=None):
        # Constructors stay side-effect free: no recv, no layer construction.
        # setup() performs the model-info exchange and builds layers.
        self.socket = socket
        self.cfg = cfg
        self.model = model
        self._inshape = inshape
        # a None engine (e.g. tests) falls back to the config-selected backend
        self.he = he if he is not None else create_he(*he_engine_args(cfg))
        # protocol params (pad_ratio / noise_sigma) passed to layer setups
        self.pro_args = cfg.server.protocol.extra_args()
        self.model_name = None
        self.model_info = None
        self.inshape = None
        # set only in run mode 'online', by select_resource()
        self.resource_id = None
        # set by prepare_protocols(); offline() requires it and clears it
        self._protocols_ready = False
        self.layers = []
        self.linears = []
        self.shortcuts = {}
        self.locals = []

    def setup(self):
        """Receive the server-parsed model info and build client layers."""
        message = recv_model_info(self.socket)
        self.model_name = message.model_name
        self.model_info = message.info
        self.inshape = self.model_info.inshape
        if self._inshape is not None and tuple(self._inshape) != tuple(self.inshape):
            raise ValueError(
                "client inshape {} does not match server model info {}".format(
                    tuple(self._inshape), tuple(self.inshape)))
        device = self.cfg.common.device
        # send the HE context once, right after the model info: the server
        # reads it before building its layers (empty for the NoHE backend)
        b_ctx = self.he.to_bytes_context()
        self.socket.sendall(struct.pack('!i', len(b_ctx)) + b_ctx)
        self.layers, self.linears, self.shortcuts, self.locals = util.make_client_model_from_info(
            self.socket, self.model_info, self.he, device, self.cfg.server.protocol.name)
        print("Model info received: {} ({} layers, input {})".format(
            self.model_name, len(self.layers), tuple(self.inshape)))

    def select_resource(self, resource_id: int = None) -> int:
        """Online mode: receive the ids available on the server, pick one
        (explicit or random) and announce it back. Must run after setup()."""
        if self.model_info is None:
            raise RuntimeError("Client.setup() must be called before select_resource()")
        available = recv_resource_ids(self.socket)
        self.resource_id = choose_index(available, resource_id)
        send_resource_id(self.socket, self.resource_id)
        print("Resource set selected: {}".format(self.resource_id))
        return self.resource_id

    def restore_resource(self, res_dir, resource_id: int) -> list:
        """Online mode: load this side's resource file and return its per-layer
        resource entries; the caller then runs prepare(resources) instead of
        drawing fresh randomness."""
        if self.model_info is None:
            raise RuntimeError("Client.setup() must be called before restore_resource()")
        resources = load_resource_set_client(self.cfg, res_dir, self.model_name, resource_id, self.model_info)
        print("Client resources restored (set {})".format(resource_id))
        return resources

    def prepare(self, resources: list = None):
        """Per-layer protocol preparation (masks and propagated state). If resources is None, draw fresh randomness"""
        if self.model_info is None:
            raise RuntimeError("Client.setup() must be called before prepare()")
        for i, lyr in enumerate(self.layers):
            print('  setup {}: {} ...'.format(i, lyr.__class__.__name__))
            if resources is not None:
                lyr.setup_res(resources[i], **self.pro_args)
            else:
                lyr.setup(**self.pro_args)
        self._protocols_ready = True

    def offline(self):
        """Per-layer offline round only; call prepare() first."""
        if self.model_info is None:
            raise RuntimeError("Client.setup() must be called before offline()")
        if not self._protocols_ready:
            raise RuntimeError("Client.prepare() must be called before offline()")
        for i, lyr in enumerate(self.layers):
            print('  offline {}: {} ...'.format(i, lyr.__class__.__name__))
            lyr.offline()
        # one resource set per prepare: the next offline() needs fresh masks
        self._protocols_ready = False

    def online(self, data):
        if self.model_info is None:
            raise RuntimeError("Client.setup() must be called before online()")
        for lyr in self.layers:
            data = lyr.online(data)
        return data