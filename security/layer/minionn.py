import sys
import time
import torch
import torch.nn as nn
import numpy as np
import copy

import ml.util
import model.minionn
# import system.util

# %% util functions

def prepare_attack_data(model, inshape, dataset, cnt, device):
    r = torch.rand(1, *inshape)

    atk_input_x = []
    atk_input_xr = []
    atk_output_x = []
    atk_output_xr = []
    loader = torch.utils.data.DataLoader(dataset, min(1000, cnt), shuffle=False)
    with torch.no_grad():
        n = 0
        for x, y in loader:
            o = model(x)
            atk_input_x.append(x)
            atk_output_x.append(o)
            xr = x - r
            o = model(xr)
            atk_input_xr.append(xr)
            atk_output_xr.append(o)
            n += x.size(0)
            if n >= cnt:
                break

    atk_input_x = torch.cat(atk_input_x)[:cnt].to(device)
    atk_input_xr = torch.cat(atk_input_xr)[:cnt].to(device)
    atk_output_x = torch.cat(atk_output_x)[:cnt]
    atk_output_xr = torch.cat(atk_output_xr)[:cnt]
    atk_target_x = (atk_output_x >= 0).float().to(device)
    atk_target_xr = (atk_output_xr >= 0).float().to(device)
    return atk_input_x, atk_input_xr, atk_target_x, atk_target_xr


def train_epoch(m_attack, loader, loss_fn):
    running_loss = 0.0
    for x, xr, y, yr in loader:
        n = len(x)
        o = m_attack(x)
        loss1 = loss_fn(o, y)
        o = m_attack(xr)
        loss2 = loss_fn(o, yr)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * n
    return running_loss

def train_epoch_cuda(m_attack, loader, loss_fn):
    running_loss = 0.0
    for x, xr, y, yr in loader:
        n = len(x)
        o = m_attack(x.cuda())
        loss1 = loss_fn(o, y.cuda())
        o = m_attack(xr.cuda())
        loss2 = loss_fn(o, yr.cuda())
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * n
    return running_loss


# %% main

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print('Usage: python minionn.py <data_dir> <known_count> [nepoch] [epsilon] [lr] [bs] [device] [load_to_gpu]')
        print('  default nepoch=1000, epsilon=1e-9, lr=0.01, bs=1000, device=cuda, load_to_gpu=True')
        exit(0)
    data_dir = args[0]
    weight_path = 'pretrained/minionn.pt'
    known_count = int(args[1])
    nepoch = int(args[2]) if len(args) > 2 else 1000
    epsilon = float(args[3]) if len(args) > 3 else 1e-9
    learning_rate = float(args[4]) if len(args) > 4 else 0.01
    batch_size = int(args[5]) if len(args) > 5 else 1000
    device = args[6] if len(args) > 6 else 'cuda'
    load_to_gpu = args[7].lower() in ['1', 'true', 'yes'] if len(args) > 7 else True

    last_idx = weight_path.rfind('.')
    fn = weight_path[weight_path.rfind('/')+1: last_idx if last_idx != -1 else len(weight_path)]

    print(f"Attack {fn} with {known_count} known data, {nepoch} epochs, {epsilon} epsilon, {learning_rate} learning rate, {batch_size} batch_size, {device} device")
    fn = f'{fn}_{known_count}'

    inshape = model.minionn.inshape
    model = model.minionn.build()
    ml.util.load_model_state(model, weight_path)
    model.eval()
    # shapes = system.util.compute_shape(model, inshape)

    dataset = ml.util.load_data('cifar10', data_dir, False, True)[1]

    # prepare attack data
    m0=model[0]
    print('Preparing attack data')
    if device != 'cpu' and load_to_gpu == False:
        atk_input_x, atk_input_xr, atk_target_x, atk_target_xr =\
            prepare_attack_data(m0, inshape, dataset, known_count, 'cpu')
    else:
        atk_input_x, atk_input_xr, atk_target_x, atk_target_xr =\
            prepare_attack_data(m0, inshape, dataset, known_count, device)

    # attack model
    m0_a = copy.deepcopy(m0)
    m_attack = nn.Sequential( m0_a, nn.Sigmoid() )
    if device != 'cpu':
        m_attack.to(device)
    for p in m_attack.parameters():
        p.data.uniform_(-1, 1)

    atk_ds = torch.utils.data.TensorDataset(
        atk_input_x, atk_input_xr, atk_target_x, atk_target_xr)
    if device != 'cpu' and load_to_gpu == False:
        loader = torch.utils.data.DataLoader(atk_ds, batch_size, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(atk_ds, batch_size)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(m_attack.parameters(), lr=learning_rate)

    # attach with ml
    m_attack.train()

    w, b = m0.weight.to(device), m0.bias.to(device)
    n_param = w.numel() + b.numel()

    print("Attacking model")

    loss_record = []
    diff_record = []
    time_record = []
    last_loss = np.inf
    delta = 2*epsilon
    for i in range(nepoch):
        if delta < epsilon:
            break
        t = time.time()
        # train with BCE loss
        if device == 'cpu':
            running_loss = train_epoch(m_attack, loader, loss_fn)
        else:
            running_loss = train_epoch_cuda(m_attack, loader, loss_fn)
        loss_record.append(running_loss)
        # compute difference with real model weights
        with torch.no_grad():
            dw = (m0_a.weight - w)/w
            diff = dw.mean().item().abs()
            # db = (m0_a.bias - b).abs()/b
            # diff = (dw + db)/n_param
        diff_record.append(dw.cpu().numpy().flatten())
        t = time.time() - t
        time_record.append(t)
        delta = np.abs(last_loss - running_loss)
        last_loss = running_loss
        eta = t * (nepoch - i)
        print(f'Epoch {i}: loss={running_loss:.6g}, impv={delta:.4g}, diff={diff:.4f}, time={t:.2f}, eta={eta:.2f}')

    fn = f'{fn}_{nepoch}'
    np.savez(fn+"_record.npz", loss_record=loss_record, diff_record=diff_record, time_record=time_record)
    # torch.save(m_attack.state_dict(), fn+'_model.pt')

