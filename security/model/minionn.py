import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

import ml.util
import model.minionn
# import system.util

# %% util functions

def prepare_attack_data(model, inshape, loader, device):
    r = torch.rand(1, *inshape)

    signs = []
    with torch.no_grad():
        for x, y in loader:
            data = x.to(device)
            for lyr in model:
                data = lyr(data)
                s = torch.sign(data)
                if isinstance(lyr, (nn.ReLU, nn.MaxPool2d)):
                    signs.append(s)
    L = len(model)
    res = [ torch.cat([s[i] for s in signs]) for i in range(L) ]
    return res

# use offline prepared sign data
def train_epoch_offline(model, dloader, sloader, loss_fn, optimizer, device):
    running_loss = 0.0
    for (x, y), signs in zip(dloader, sloader):
        x.to(device)
        y.to(device)
        n = len(x)
        slist = []
        data = x
        for lyr in model:
            data = lyr(data)
            s = torch.sign(data)
            slist.append(s)
        slist = torch.cat(slist)
        loss = loss_fn(data, y, slist, signs)
        # ploss = ploss_fn(data, y)
        # sloss = sloss_fn(slist, signs)
        # loss = ploss + sloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * n
    return running_loss


# generate sign data online
def train_epoch_online(model, model_ref, loader, loss_fn, optimizer, device):
    running_loss = 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        n = len(x)
        datar = x
        datat = x.clone()
        sign_r = []
        sign_t = []
        for lyr_r, lyr_t in zip(model_ref, model):
            with torch.no_grad():
                datar = lyr_r(datar)
            datat = lyr_t(datat)
            if isinstance(lyr_r, (nn.ReLU, nn.MaxPool2d)):
                sign_r.append(torch.sign(datar).view(-1))
                sign_t.append(torch.sign(datat).view(-1))
        loss = loss_fn(datat, y, torch.concat(sign_t), torch.concat(sign_r))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * n
    return running_loss


def make_loss_function(theta):
    factor = theta
    def loss_fn(pred, target, signs, signs_ref):
        ploss = F.cross_entropy(pred, target)
        sloss = F.mse_loss(signs, signs_ref)
        return ploss + factor * sloss
    return loss_fn

# %% main

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print('Usage: python minionn.py <data_dir> <known_count> [nepoch] [epsilon] [lr] [bs] [device] [rnd] [offline_gen]')
        print('  default: nepoch=1000, epsilon=1e-9, lr=0.01, bs=1000, device=cuda, rnd=False, offline_gen=True')
        exit(0)
    data_dir = args[0]
    weight_path = 'pretrained/minionn.pt'
    known_count = int(args[1])
    nepoch = int(args[2]) if len(args) > 2 else 1000
    epsilon = float(args[3]) if len(args) > 3 else 1e-9
    learning_rate = float(args[4]) if len(args) > 4 else 0.01
    batch_size = int(args[5]) if len(args) > 5 else 1000
    device = args[6] if len(args) > 6 else 'cuda'
    rnd = args[7].lower() in ['1', 'true', 't', 'yes', 'y'] if len(args) > 7 else False
    offline_gen = args[8].lower() in ['1', 'true', 't', 'yes', 'y'] if len(args) > 8 else True

    last_idx = weight_path.rfind('.')
    fn = weight_path[weight_path.rfind('/')+1: last_idx if last_idx != -1 else len(weight_path)]

    print(f"Attack {fn} with {known_count} known data, {nepoch} epochs, {epsilon} epsilon, {learning_rate} learning rate, {batch_size} batch_size, {device} device, {offline_gen} offline_gen")
    fn = f'{fn}_{known_count}'

    # 1. prepare models
    # 1.1 reference model
    inshape = model.minionn.inshape
    m_ref = model.minionn.build()
    ml.util.load_model_state(m_ref, weight_path)
    m_ref.eval()
    # shapes = system.util.compute_shape(model, inshape)
    
    # 1.2 attack model
    m = copy.deepcopy(m_ref)
    if device != 'cpu':
        m.to(device)
    for p in m.parameters():
        p.data.uniform_(-1, 1)

    # 2. prepare data
    # 2.1 prepare trainset
    print('Loading sample data')
    trainset, testset = ml.util.load_data('cifar10', data_dir, True, True)
    if known_count < len(trainset):
        trainset = [trainset[i] for i in range(known_count)]
    shf_load = offline_gen == False
    pin_load = device != 'cpu'
    loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=shf_load, pin_memory=pin_load)

    # 2.2 prepare attack data
    print('Preparing attack data')
    if offline_gen:
        print('Generating attack data offline')
        signs = prepare_attack_data(m_ref, inshape, loader, device)
    else:
        print('Skip: will generate attack data online')

    # 3. attach with ml
    print("Attacking model")

    theta = 1.0
    loss_fn = make_loss_function(theta)

    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    m.train()

    loss_record = []
    time_record = []
    last_loss = np.inf
    delta = 2*epsilon
    delta_ema = delta
    t0 = time.time()
    for i in range(nepoch):
        if delta_ema < epsilon:
            break
        t1 = time.time()
        # train with BCE loss
        if offline_gen:
            running_loss = train_epoch_offline(m, m_ref, signs, loss_fn, optimizer, device)
        else:
            running_loss = train_epoch_online(m, m_ref, loader, loss_fn, optimizer, device)
        loss_record.append(running_loss)

        t = time.time() - t1
        time_record.append(t)
        # moving average for delta (less noise affect)
        delta = np.abs(last_loss - running_loss)
        delat_ema = delta_ema * 0.4 + delta * 0.6
        last_loss = running_loss
        if i % 50 == 49:
            eta = (t1-t0) / (i+1) * (nepoch - i)
            print(f'Epoch {i}: loss={running_loss:.6g}, impv={delta:.4g}, time={t:.2f}, {t1-t0:.2f}, eta={eta:.2f}')

    fn = f'{fn}_{nepoch}'
    np.savez(fn+"_record.npz", loss_record=loss_record, time_record=time_record)
    # torch.save(m_attack.state_dict(), fn+'_model.pt')

