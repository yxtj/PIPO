import socket
import struct
import torch
import torch.nn as nn

from src.comm.model_info import send_model_info
from src.comm.resource import recv_resource_id, send_resource_ids
from src.common.config import he_engine_args
from src.common.model_info import build_model_info
from src.heutil import create_he
from src.system import util
from src.system.resource import list_indices, load_resource_set_server


class Server():
    def __init__(self, socket: socket.socket, model: nn.Module, inshape: tuple,
                 cfg, model_name: str = None):
        # Constructors stay side-effect free: no I/O, no shape analysis, no
        # layer construction. setup() does all of that after construction.
        self.socket = socket
        self.model = model
        self.inshape = inshape
        self.cfg = cfg
        self.model_name = model_name
        self.model_info = None
        self.he = None
        self.layers = []
        self.linears = []
        self.shortcuts = {}
        self.locals = []
        self.dependency = {}
        # protocol params (pad_ratio / noise_sigma) passed to layer setups
        self.pro_args = cfg.server.protocol.extra_args()

    def setup(self):
        """Parse the model, send its info to the client, and build server layers."""
        if self.model_name is None:
            self.model_name = self.cfg.common.model.name
        self.model_info = build_model_info(self.model, self.inshape)
        send_model_info(self.socket, self.model_name, self.model_info)
        print("Model info sent: {} ({} layers, input {})".format(
            self.model_name, len(self.model_info.steps), tuple(self.inshape)))
        # one shared engine for all layers; the client sends its HE context
        # once, right after receiving the model info (empty for NoHE)
        self.he = create_he(*he_engine_args(self.cfg))
        ctx_len = struct.unpack('!i', self.socket.recv(4))[0]
        b_ctx = self.socket.recv(ctx_len)
        self.he.from_bytes_context(b_ctx)
        self.layers, self.linears, self.shortcuts, self.locals = \
            util.make_server_model(self.socket, self.model, self.inshape, self.he,
                                   self.cfg.common.device, self.cfg.server.protocol.name)
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

    def select_resource(self, res_dir, resource_id: int = None) -> int:
        """Online mode: announce the available resource-set ids and receive
        the one the client picked. Must run after setup()."""
        if self.model_info is None:
            raise RuntimeError("Server.setup() must be called before select_resource()")
        available = list_indices(res_dir, self.model_name)
        if not available:
            raise FileNotFoundError(
                "no resource sets for model {!r} under {}".format(self.model_name, res_dir))
        if resource_id is not None and resource_id not in available:
            raise FileNotFoundError(
                "resource set {} not available for model {!r} (have: {})".format(
                    resource_id, self.model_name, available))
        send_resource_ids(self.socket, available)
        chosen = recv_resource_id(self.socket)
        if chosen not in available:
            raise ValueError(
                "client selected resource set {}, which is outside the announced range {}".format(
                    chosen, available))
        print("Resource set selected: {} (available: {})".format(chosen, available))
        return chosen

    def restore_resource(self, res_dir, resource_id: int) -> None:
        """Online mode: load this side's resource file and set the layers up
        from it instead of drawing fresh randomness."""
        if self.model_info is None:
            raise RuntimeError("Server.setup() must be called before restore()")
        entries = load_resource_set_server(self.cfg, res_dir, self.model_name, resource_id, self.model_info)
        self.setup_layers(res=entries)
        print("Server resources restored (set {})".format(resource_id))

    def setup_layers(self, res: list = None):
        """Per-layer protocol setup. With res (the layer entries of a loaded
        resource set, run mode 'online') the persisted masks/shuffles are
        passed through setup kwargs instead of fresh randomness; the layer
        order matters, because each layer's setup derives its propagated
        state (mlast / rplast / kin) from the previous, already-restored
        layer's protocol."""
        if self.model_info is None:
            raise RuntimeError("Server.setup() must be called before setup_layers()")
        last_non_local = util.find_last_non_local_layer(len(self.layers), self.locals)
        last_lyr = None
        for i, lyr in enumerate(self.layers):
            name = lyr.__class__.__name__
            print('  setup {}: {}(inshape={}, outshape={}) ...'.format(i, name, lyr.ishape, lyr.oshape))
            # the last non-local layer hands its plaintext result to the
            # client, so the protocol must not mask or permute its output
            if res is not None:
                lyr.setup_res(last_lyr, is_last_layer=(i == last_non_local), entry=res[i],
                              **self.pro_args)
            else:
                lyr.setup(last_lyr, is_last_layer=(i == last_non_local), **self.pro_args)
            last_lyr = lyr

    def do_offline(self):
        """Interactive offline phase (assumes setup_layers() has run)."""
        data = None
        for i, lyr in enumerate(self.layers):
            data = lyr.offline() # get the input of this layer (i-th intermediate result)
            # handle shortcut connections: update the input buffer of shortcut-end layer
            if i in self.dependency:
                for j in self.dependency[i]:
                    if j != i: # the own input fm is passed directly to the layer itself
                        self.layers[j].update_offline(i, data)

    def offline(self):
        """Standard offline phase: per-layer setup interleaved with the
        interactive offline round. May be called repeatedly (one resource set
        per call)."""
        if self.model_info is None:
            raise RuntimeError("Server.setup() must be called before offline()")
        last_non_local = util.find_last_non_local_layer(len(self.layers), self.locals)
        last_lyr = None
        data = None
        for i, lyr in enumerate(self.layers):
            name = lyr.__class__.__name__
            print('  offline {}: {}(inshape={}, outshape={}) ...'.format(i, name, lyr.ishape, lyr.oshape))
            # setup
            lyr.setup(last_lyr, is_last_layer=(i == last_non_local), **self.pro_args)
            last_lyr = lyr
            # offline
            data = lyr.offline() # get the input of this layer (i-th intermediate result)
            # handle shortcut connections: update the input buffer of shortcut-end layer
            if i in self.dependency:
                for j in self.dependency[i]:
                    if j != i: # the own input fm is passed directly to the layer itself
                        self.layers[j].update_offline(i, data)

    def online(self):
        if self.model_info is None:
            raise RuntimeError("Server.setup() must be called before online()")
        for i, lyr in enumerate(self.layers):
            data = lyr.online() # get the input of this layer (i-th intermediate result)
            if i in self.dependency:
                for j in self.dependency[i]:
                    if j != i:
                        self.layers[j].update_online(i, data)