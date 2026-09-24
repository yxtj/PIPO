import torch
import torch.nn as nn

from src.layer_basic.stat import Stat
from src.common.model_info import compute_shapes
from src.model.dag_model import AddOp, ConcatOp, DagModel, JumpOp

import src.layer as layer


def flat_layers(model):
    """Flatten a (possibly nested) DagModel into its leaf execution list.

    Returns (modules, sources) where `sources[i]` is the tuple of feature-map
    indices the i-th leaf reads from. For a plain nn.Sequential each layer
    simply reads the previous result. A leaf writing FM k+1 gets global FM
    numbering, so the i-th leaf's input/output are FM i / FM i+1 - the same
    numbering used by the runtime per-intermediate-result plumbing.
    """
    if isinstance(model, DagModel):
        entries = [e for e in model.flat_entries() if not e.is_block]
        return [e.module for e in entries], [list(e.sources) for e in entries]
    if isinstance(model, nn.Sequential):
        return list(model), [[i] for i in range(len(model))]
    raise Exception("Model should be either Sequential or DagModel.")


def compute_shape(model, inshape):
    """Feature-map shapes (batched): FM 0 = input, FM i+1 = output of the i-th leaf.

    The runtime tensors carry a batch dimension, so each feature-map shape
    gets a leading 1.
    """
    return [(1,) + s for s in compute_shapes(model, inshape)]


def make_client_model(socket, model, inshape, he):
    modules, sources = flat_layers(model)
    shapes = compute_shape(model, inshape)
    device = next(model.parameters()).device if next(model.parameters(), None) is not None else "cpu"
    layers = []
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(modules):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.ConvClient(socket, shapes[i], shapes[i+1], he, device))
            linears.append(i)
        elif isinstance(lyr, nn.Linear):
            layers.append(layer.FcClient(socket, shapes[i], shapes[i+1], he, device))
            linears.append(i)
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.ReLUClient(socket, shapes[i], shapes[i+1], he, device))
            locals.append(i)
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.MaxPoolClient(socket, shapes[i], shapes[i+1], he, lyr, device))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.AvgPoolClient(socket, shapes[i], shapes[i+1], he, lyr, device))
            linears.append(i)
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.FlattenClient(socket, shapes[i], shapes[i+1], he, device))
            locals.append(i)
            # the input of the flatten layer should have a concrete shape
        elif isinstance(lyr, JumpOp):
            layers.append(layer.JumpClient(socket, shapes[i], shapes[i+1], he, device, srcs=sources[i], own=i))
            scl[i] = list(sources[i])
        elif isinstance(lyr, AddOp):
            layers.append(layer.AdditionClient(socket, shapes[i], shapes[i+1], he, device, srcs=sources[i], own=i))
            scl[i] = list(sources[i])
        elif isinstance(lyr, ConcatOp):
            layers.append(layer.ConcatenationClient(socket, shapes[i], shapes[i+1], he, device, srcs=sources[i], own=i))
            scl[i] = list(sources[i])
        elif isinstance(lyr, nn.Identity):
            layers.append(layer.IdentityClient(socket, shapes[i], shapes[i+1], he, device))
            linears.append(i)
        elif isinstance(lyr, nn.Softmax):
            assert i == len(modules) - 1, "Softmax should be the last layer."
            layers.append(layer.SoftmaxClient(socket, shapes[i], shapes[i+1], he, device))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input fm idx list}
    for idx, oidx in scl.items():
        if any(isinstance(layers[i], layer.LocalLayerClient) for i in oidx):
            msg = "Shortcut {} input should not be a local layer. Checking the model or adding an identity layer.".format(idx)
            raise Exception(msg)
        shortcuts[idx] = oidx
    # shortcuts is {shortcut layer idx: intermediate result idx}
    return layers, linears, shortcuts, locals


def make_server_model(socket, model, inshape):
    modules, sources = flat_layers(model)
    shapes = compute_shape(model, inshape)
    device = next(model.parameters()).device if next(model.parameters(), None) is not None else "cpu"
    layers = []
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(modules):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.ConvServer(socket, shapes[i], shapes[i+1], lyr, device))
            linears.append(i)
        elif isinstance(lyr, nn.Linear):
            layers.append(layer.FcServer(socket, shapes[i], shapes[i+1], lyr, device))
            linears.append(i)
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.ReLUServer(socket, shapes[i], shapes[i+1], lyr, device))
            locals.append(i)
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.MaxPoolServer(socket, shapes[i], shapes[i+1], lyr, device))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.AvgPoolServer(socket, shapes[i], shapes[i+1], lyr, device))
            linears.append(i)
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.FlattenServer(socket, shapes[i], shapes[i+1], lyr, device))
            locals.append(i)
        elif isinstance(lyr, JumpOp):
            layers.append(layer.JumpServer(socket, shapes[i], shapes[i+1], lyr, device, srcs=sources[i], own=i))
            scl[i] = list(sources[i])
        elif isinstance(lyr, AddOp):
            layers.append(layer.AdditionServer(socket, shapes[i], shapes[i+1], lyr, device, srcs=sources[i], own=i))
            scl[i] = list(sources[i])
        elif isinstance(lyr, ConcatOp):
            layers.append(layer.ConcatenationServer(socket, shapes[i], shapes[i+1], lyr, device, srcs=sources[i], own=i))
            scl[i] = list(sources[i])
        elif isinstance(lyr, nn.Identity):
            layers.append(layer.IdentityServer(socket, shapes[i], shapes[i+1], lyr, device))
            linears.append(i)
        elif isinstance(lyr, nn.Softmax):
            assert i == len(modules) - 1, "Softmax should be the last layer."
            layers.append(layer.SoftmaxServer(socket, shapes[i], shapes[i+1], lyr, device))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input fm idx list}
    for idx, oidx in scl.items():
        if any(isinstance(layers[i], layer.LocalLayerClient) for i in oidx):
            msg = "Shortcut {} input should not be a local layer. Checking the model or adding an identity layer.".format(idx)
            raise Exception(msg)
        shortcuts[idx] = oidx
    # shortcuts is {shortcut layer idx: intermediate result idx}
    return layers, linears, shortcuts, locals


def find_last_non_local_layer(num_layer, local_layers):
    for i in range(num_layer-1, -1, -1):
        if i not in local_layers:
            return i
    return -1


def analyze_stat(layers, n):
    s_total = Stat()
    s_relu = Stat()
    s_linear = Stat()
    s_l_conv = Stat()
    s_l_fc = Stat()
    s_pool = Stat()
    s_sc = Stat()
    for i, lyr in enumerate(layers):
        print("  Layer {} {}: {}".format(i, lyr.__class__.__name__, lyr.stat))
        s_total += lyr.stat
        if isinstance(lyr, (layer.ReLUServer, layer.ReLUClient)):
            s_relu += lyr.stat
        elif isinstance(lyr, (layer.MaxPoolServer, layer.MaxPoolClient,
                              layer.AvgPoolServer, layer.AvgPoolClient)):
            s_pool += lyr.stat
        elif isinstance(lyr, (layer.ConvServer, layer.ConvClient,
                              layer.FcServer, layer.FcClient)):
            s_linear += lyr.stat
            if isinstance(lyr, (layer.ConvServer, layer.ConvClient)):
                s_l_conv += lyr.stat
            else:
                s_l_fc += lyr.stat
        elif isinstance(lyr, (layer.ShortCutServer, layer.ShortCutClient)):
            s_sc += lyr.stat
    return s_total, s_relu, s_linear, s_l_conv, s_l_fc, s_pool, s_sc
