import torch
import torch.nn as nn
import torch_extension as te

from layer_basic.stat import Stat


import layer

def compute_shape(model, inshape):
    # TODO: optimize this function to remove the dummy execution of the model
    # if len(inshape) == 3 or len(inshape) == 1:
    inshape = (1, *inshape)
    t = torch.zeros(inshape)
    if next(model.parameters()).is_cuda:
        t = t.cuda()
    shapes = [inshape]
    if isinstance(model, te.SequentialShortcut):
        for i, lyr in enumerate(model):
            # print(i, lyr, tuple(t.shape))
            if i in model.dependency:
                for j in model.dependency[i]:
                    model[j].update(i-j-1, t)
            t = lyr(t)
            shapes.append(tuple(t.shape))
    elif isinstance(model, nn.Sequential):
        for i, lyr in enumerate(model):
            t = lyr(t)
            shapes.append(tuple(t.shape))
    else:
        raise Exception("Model should be either Sequential or SequentialShortcut.")
    return shapes

    
def make_client_model(socket, model, inshape, he):
    device = next(model.parameters()).device
    shapes = compute_shape(model, inshape)
    layers = []
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(model):
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
        elif isinstance(lyr, te.Jump):
            layers.append(layer.JumpClient(socket, shapes[i], shapes[i+1], he, device))
            scl[i] = [i + r for r in lyr.relOther]
        elif isinstance(lyr, te.Addition):
            layers.append(layer.AdditionClient(socket, shapes[i], shapes[i+1], he, device))
            scl[i] = [i + r for r in lyr.relOther]
        elif isinstance(lyr, te.Concatenation):
            layers.append(layer.ConcatenationClient(socket, shapes[i], shapes[i+1], he, device))
            scl[i] = [i + r for r in lyr.relOther]
        elif isinstance(lyr, nn.Identity):
            layers.append(layer.IdentityClient(socket, shapes[i], shapes[i+1], he, device))
            linears.append(i)
        elif isinstance(lyr, nn.Softmax):
            assert i == len(model) - 1, "Softmax should be the last layer."
            layers.append(layer.SoftmaxClient(socket, shapes[i], shapes[i+1], he, device))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input layer idx}
    for idx, oidx in scl.items():
        oidx = [i + 1 for i in oidx] # move to the output of the layer
        if any(isinstance(layers[i], layer.LocalLayerClient) for i in oidx):
            msg = "Shortcut {} input should not be a local layer. Checking the model or adding an identity layer.".format(idx)
            raise Exception(msg)
        shortcuts[idx] = oidx
    # shortcuts is {shortcut layer idx: intermediate result idx}
    return layers, linears, shortcuts, locals


def make_server_model(socket, model, inshape):
    device = next(model.parameters()).device
    shapes = compute_shape(model, inshape)
    layers = []
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(model):
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
        elif isinstance(lyr, te.Jump):
            layers.append(layer.JumpServer(socket, shapes[i], shapes[i+1], lyr, device))
            scl[i] = [i + r for r in lyr.relOther]
        elif isinstance(lyr, te.Addition):
            layers.append(layer.AdditionServer(socket, shapes[i], shapes[i+1], lyr, device))
            scl[i] = [i + r for r in lyr.relOther]
        elif isinstance(lyr, te.Concatenation):
            layers.append(layer.ConcatenationServer(socket, shapes[i], shapes[i+1], lyr, device))
            scl[i] = [i + r for r in lyr.relOther]
        elif isinstance(lyr, nn.Identity):
            layers.append(layer.IdentityServer(socket, shapes[i], shapes[i+1], lyr, device))
            linears.append(i)
        elif isinstance(lyr, nn.Softmax):
            assert i == len(model) - 1, "Softmax should be the last layer."
            layers.append(layer.SoftmaxServer(socket, shapes[i], shapes[i+1], lyr, device))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input layer idx list}
    for idx, oidx in scl.items():
        oidx = [i + 1 for i in oidx] # move to the output of the layer
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
