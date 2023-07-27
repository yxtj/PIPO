# reference: https://github.com/Hzzone/pytorch-openpose/blob/master/src/body.py

import torch
import torch.nn as nn
import torch_extension as te
from . import op_impl

# __ALL__ = [inshape, build, padRightDownCorner]

inshape = (3, 368, 368)
# inshape = (3, 36, 36)

# transfer caffe model to pytorch which will match the layer name
def load_model(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    model.load_state_dict(transfered_model_weights)
    return model


def build_body_model(weight_path=None):
    m = op_impl.bodypose_model()
    if weight_path is not None:
        load_model(m, torch.load(weight_path))

    ln = [
        (len(m.model0), 0),
        (len(m.model1_1), len(m.model1_2)),
        (len(m.model2_1), len(m.model2_2)),
        (len(m.model3_1), len(m.model3_2)),
        (len(m.model4_1), len(m.model4_2)),
        (len(m.model5_1), len(m.model5_2)),
        (len(m.model6_1), len(m.model6_2)),
    ]
    presum = [ 0, ]
    for i in range(1, len(ln)):
        presum.append(presum[i-1] + ln[i][0] + ln[i][1] + 2)

    model = te.SequentialShortcut(
        *m.model0,
        *m.model1_1, te.Jump(-ln[1][0]-1), *m.model1_2, te.Concatenation([-ln[1][1]-2, -presum[1]], 1, order=[-ln[1][1]-2, -1, -presum[1]]),
        *m.model2_1, te.Jump(-ln[2][0]-1), *m.model2_2, te.Concatenation([-ln[2][1]-2, -presum[2]], 1, order=[-ln[2][1]-2, -1, -presum[2]]),
        *m.model3_1, te.Jump(-ln[3][0]-1), *m.model3_2, te.Concatenation([-ln[3][1]-2, -presum[3]], 1, order=[-ln[3][1]-2, -1, -presum[3]]),
        *m.model4_1, te.Jump(-ln[4][0]-1), *m.model4_2, te.Concatenation([-ln[4][1]-2, -presum[4]], 1, order=[-ln[4][1]-2, -1, -presum[4]]),
        *m.model5_1, te.Jump(-ln[5][0]-1), *m.model5_2, te.Concatenation([-ln[5][1]-2, -presum[5]], 1, order=[-ln[5][1]-2, -1, -presum[5]]),
        *m.model6_1, te.Jump(-ln[6][0]-1), *m.model6_2, te.Concatenation([-ln[6][1]-2], 1, order=[-ln[6][1]-2, -1]),
    )
    return model

def build_hand_model(weight_path=None):
    m = op_impl.handpose_model()
    if weight_path is not None:
        load_model(m, torch.load(weight_path))

    ln = [
        len(m.model1_0), len(m.model1_1),
        len(m.model2), len(m.model3), len(m.model4), len(m.model5), len(m.model6),
    ]
    presum = [ 0, ]
    for i in range(1, len(ln)):
        presum.append(presum[i-1] + ln[i] + 1)

    model = te.SequentialShortcut(
        *m.model1_0,
        *m.model1_1, te.Concatenation(-presum[1], 1, [-1, -presum[1]]),
        *m.model2, te.Concatenation(-presum[2], 1, [-1, -presum[2]]),
        *m.model3, te.Concatenation(-presum[3], 1, [-1, -presum[3]]),
        *m.model4, te.Concatenation(-presum[4], 1, [-1, -presum[4]]),
        *m.model5, te.Concatenation(-presum[5], 1, [-1, -presum[5]]),
        *m.model6,
    )
    return model

## API function

def build(model: str, weight_path=None):
    assert model in ['body', 'hand']
    if model == 'body':
        return build_body_model(weight_path)
    else:
        return build_hand_model(weight_path)


def padRightDownCorner(img, stride, padValue, square=False):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right
    if square:
        if h > w:
            pad[3] = h + pad[2] - w
        else:
            pad[2] = w + pad[3] - h

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def sep_body_result(result):
    heatmap = result[:, :38]
    paf = result[:, 38:]
    return heatmap, paf
