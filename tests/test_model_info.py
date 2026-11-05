# -*- coding: utf-8 -*-
"""Server-side model parsing -> client-side specs, without a local model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch.nn as nn

from models import registry
from src.common.config import load_config, set_config
from src.common.model_info import build_model_info
from src.common.model_info_io import model_info_from_json, model_info_hash, model_info_to_json
from src.common.model_spec import build_client_module, build_client_specs
from src.model.dag_model import AddOp, ConcatOp, JumpOp
from src.system import Client, Server, util

set_config(load_config())

MODELS = [
    "poc-0-linear",
    "poc-0-max",
    "poc-0-avg",
    "poc-0-add",
    "poc-0-con",
    "poc-0-jmp",
    "poc-1",
    "poc-2",
    "poc-5",
    "poc-6",
    "poc-7",
    "poc-8",
    "poc-9-leakyrelu",
    "poc-10-sigmoid",
    "poc-10-gelu",
    "minionn",
]

LOCAL_OPS = (nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.GELU, nn.Flatten, nn.Softmax)
SHORTCUT_OPS = (AddOp, ConcatOp, JumpOp)


@pytest.mark.parametrize("name", MODELS)
def test_specs_match_model(name):
    inshape, model = registry.load(name, seed=0)
    info = model_info_from_json(model_info_to_json(build_model_info(model, inshape)))
    specs = build_client_specs(info)
    modules, _ = util.flat_layers(model)
    shapes = util.compute_shape(model, inshape)

    assert len(specs) == len(modules)
    for i, (spec, module) in enumerate(zip(specs, modules)):
        assert spec.index == i
        assert spec.op == type(module).__name__.lower()
        assert (1, *spec.ishape) == shapes[i]
        assert (1, *spec.oshape) == shapes[i + 1]
        if isinstance(module, LOCAL_OPS):
            assert spec.kind == "local"
        elif isinstance(module, nn.MaxPool2d):
            assert spec.kind == "pool"
        elif isinstance(module, SHORTCUT_OPS):
            assert spec.kind == "shortcut"
        else:
            assert spec.kind == "linear"


def test_make_client_model_from_info_matches_wrapper():
    inshape, model = registry.load("poc-6", seed=0)
    info = build_model_info(model, inshape)
    specs = build_client_specs(info)
    layers, linears, shortcuts, locals_ = util.make_client_model_from_info(None, info, None, "cpu")
    legacy = util.make_client_model(None, model, inshape, None)

    assert [type(x).__name__ for x in layers] == [type(x).__name__ for x in legacy[0]]
    assert linears == legacy[1]
    assert shortcuts == legacy[2]
    assert locals_ == legacy[3]
    assert len(layers) == len(specs)


def test_module_attrs_roundtrip():
    cases = [
        ("poc-9-leakyrelu", "leakyrelu", nn.LeakyReLU),
        ("poc-10-gelu", "gelu", nn.GELU),
        ("poc-2", "maxpool2d", nn.MaxPool2d),
        ("poc-2-avg", "avgpool2d", nn.AvgPool2d),
        ("poc-1", "flatten", nn.Flatten),
    ]
    for name, op, cls in cases:
        inshape, model = registry.load(name, seed=0)
        specs = build_client_specs(build_model_info(model, inshape))
        spec = next(s for s in specs if s.op == op)
        module = build_client_module(spec)
        assert isinstance(module, cls), (name, op, type(module).__name__)

    inshape, model = registry.load("poc-9-leakyrelu", seed=0)
    specs = build_client_specs(build_model_info(model, inshape))
    leaky = next(s for s in specs if s.op == "leakyrelu")
    assert build_client_module(leaky).negative_slope == pytest.approx(0.2)


def test_constructors_do_no_io():
    class BoomSocket:
        def sendall(self, data):
            raise AssertionError("constructor performed I/O")

        def recv(self, n):
            raise AssertionError("constructor performed I/O")

    inshape, model = registry.load("poc-1", seed=0)
    client = Client(BoomSocket())
    assert client.info is None and client.layers == []
    server = Server(BoomSocket(), model, inshape)
    assert server.info is None and server.layers == []


def test_model_info_wire_roundtrip():
    from src.comm.model_info import recv_model_info, send_model_info

    inshape, model = registry.load("poc-6", seed=0)
    info = build_model_info(model, inshape)
    # in-memory duplex channels: the sandbox blocks loopback TCP, and the wire
    # helpers only need sendall/recv
    server_to_client = _MemoryPipe()
    client_to_server = _MemoryPipe()
    sender = _MemorySocket(in_pipe=client_to_server, out_pipe=server_to_client)
    receiver = _MemorySocket(in_pipe=server_to_client, out_pipe=client_to_server)

    send_model_info(sender, "poc-6", info)
    message = recv_model_info(receiver)

    assert message.model_name == "poc-6"
    assert model_info_hash(message.info) == model_info_hash(info)


class _MemoryPipe:
    def __init__(self):
        self._buffer = bytearray()

    def sendall(self, data):
        self._buffer.extend(data)

    def recv(self, n):
        data = bytes(self._buffer[:n])
        del self._buffer[:n]
        return data


class _MemorySocket:
    def __init__(self, in_pipe, out_pipe):
        self._in = in_pipe
        self._out = out_pipe

    def sendall(self, data):
        self._out.sendall(data)

    def recv(self, n):
        return self._in.recv(n)