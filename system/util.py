import torch
import torch.nn as nn
import torch_extension.shortcut as te

from layer_basic.stat import Stat


# from setting import PROTOCOL
# if PROTOCOL == 'dual':
#     import layer_dual as layer
# else:
#     import layer_smp as layer
import layer

def compute_shape(model, inshape):
    if len(inshape) == 3:
        inshape = (1, *inshape)
    t = torch.zeros(inshape)
    shapes = [inshape]
    for i, lyr in enumerate(model):
        t = lyr(t)
        shapes.append(tuple(t.shape))
    return shapes

    
def make_client_model(socket, model, inshape, he):
    shapes = compute_shape(model, inshape)
    layers = []
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.ConvClient(socket, shapes[i], shapes[i+1], he))
            linears.append(i)
        elif isinstance(lyr, nn.Linear):
            layers.append(layer.FcClient(socket, shapes[i], shapes[i+1], he))
            linears.append(i)
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.ReLUClient(socket, shapes[i], shapes[i+1], he))
            locals.append(i)
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.MaxPoolClient(socket, shapes[i], shapes[i+1], he, lyr))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.AvgPoolClient(socket, shapes[i], shapes[i+1], he, lyr))
            linears.append(i)
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.FlattenClient(socket, shapes[i], shapes[i+1], he))
            locals.append(i)
        elif isinstance(lyr, te.ShortCut):
            layers.append(layer.ShortCutClient(socket, shapes[i], shapes[i+1], he))
            idx = i + lyr.otherlayer # lyr.otherlayer is a negative index
            assert not isinstance(model[idx], layer.LocalLayerClient),\
                "Shortcut input should not be a local layer. Checking the model or adding an identity layer."
            scl[i] = idx
        elif isinstance(lyr, nn.Identity):
            layers.append(layer.IdentityClient(socket, shapes[i], shapes[i+1], he))
            linears.append(i)
        elif isinstance(lyr, nn.Softmax):
            assert i == len(model) - 1, "Softmax should be the last layer."
            layers.append(layer.SoftmaxClient(socket, shapes[i], shapes[i+1], he))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input layer idx}
    for idx, oidx in scl.items():
        oidx += 1 # move to the outputo of the layer
        if isinstance(layers[oidx], layer.LocalLayerClient):
            raise Exception("Shortcut input should not be a local layer.")
        shortcuts[idx] = oidx
    return layers, linears, shortcuts, locals


def make_server_model(socket, model, inshape):
    shapes = compute_shape(model, inshape)
    layers = []
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.ConvServer(socket, shapes[i], shapes[i+1], lyr))
            linears.append(i)
        elif isinstance(lyr, nn.Linear):
            layers.append(layer.FcServer(socket, shapes[i], shapes[i+1], lyr))
            linears.append(i)
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.ReLUServer(socket, shapes[i], shapes[i+1], lyr))
            locals.append(i)
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.MaxPoolServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.AvgPoolServer(socket, shapes[i], shapes[i+1], lyr))
            linears.append(i)
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.FlattenServer(socket, shapes[i], shapes[i+1], lyr))
            locals.append(i)
        elif isinstance(lyr, te.ShortCut):
            layers.append(layer.ShortCutServer(socket, shapes[i], shapes[i+1], lyr))
            idx = i + lyr.otherlayer # lyr.otherlayer is a negative index
            assert not isinstance(model[idx], layer.LocalLayerServer),\
                "Shortcut input should not be a local layer. Checking the model or adding an identity layer."
            scl[i] = idx
        elif isinstance(lyr, nn.Identity):
            layers.append(layer.IdentityServer(socket, shapes[i], shapes[i+1], lyr))
            linears.append(i)
        elif isinstance(lyr, nn.Softmax):
            assert i == len(model) - 1, "Softmax should be the last layer."
            layers.append(layer.SoftmaxServer(socket, shapes[i], shapes[i+1], lyr))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input layer idx}
    for idx, oidx in scl.items():
        oidx += 1 # move to the outputo of the layer
        if isinstance(layers[oidx], layer.LocalLayerServer):
            raise Exception("Shortcut input should not be a local layer.")
        shortcuts[idx] = oidx
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
    