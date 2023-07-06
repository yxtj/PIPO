import torch
import torch.nn as nn
import sys
import re

import model.resnet as resnet
import ml.util as util


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: python resnet.py data_dir chkpt_dir [epochs] [dump_interval] [bs] [lr] [device] [model_version]")
        print("  default epochs: 100, dump_interval: 10, bs: 512, lr: 0.001, device: cuda, model_version: 32-3")
        print("  model_version: <depth>-<version>[d], where depth is 20, 32, 44, 56, 110, 156; version is 1, 2, 3, 4; 'd' indicates direct/residual")
        sys.exit(1)
    data_dir = argv[0] # 'E:/Data/CIFAR100'
    chkpt_dir = argv[1] # 'pretrained/'
    epochs = int(argv[2]) if len(argv) > 2 else 100
    dump_interval = int(argv[3]) if len(argv) > 3 else 10
    batch_size = int(argv[4]) if len(argv) > 4 else 512
    learning_rate = float(argv[5]) if len(argv) > 5 else 0.001
    device = argv[6] if len(argv) > 6 else 'cuda'
    model_version = argv[7] if len(argv) > 7 else "3"
    
    m = re.match(r'(\d+)-(\d)([db]*)', model_version)
    if m is None:
        print("Invalid model version: {}".format(model_version))
        sys.exit(1)
    depth = int(m.group(1))
    version = int(m.group(2))
    residual = 'd' in m.group(3)
    batch_norm = 'b' in m.group(3)
    
    prefix= f'resnet{model_version}_'
    model = resnet.build(depth, version, residual, batch_norm)
    
    trainset, testset = util.load_data('cifar100', data_dir, True, True)

    file, epn = util.find_latest_model(chkpt_dir, prefix)
    if file is None:
        print("No pretrained model found")
        acc = 0.0
    else:
        print("Loading model from {}".format(file))
        util.load_model_state(model, file)
        acc = util.test(model, testset, batch_size, device=device)
        print("  Accuracy of loaded model: {:.2f}%".format(100*acc))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    util.process(model, trainset, testset, batch_size, epochs, optimizer, loss_fn,
                 dump_interval, chkpt_dir, prefix, epn, acc, device)
