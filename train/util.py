import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# data load

def load_data(name: str, folder: str, train: bool = True, test: bool = False):
    if name.lower() == 'cifar10':
        dsc = torchvision.datasets.CIFAR10
        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        ])
    elif name.lower() == 'cifar100':
        dsc = torchvision.datasets.CIFAR100
        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
        ])
    elif name.lower() == 'mnist':
        dsc = torchvision.datasets.MNIST
        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])
    # load data
    dataset_train = dsc(root=folder, train=True, download=True, transform=tsfm) if train else None
    dataset_test = dsc(root=folder, train=False, download=True, transform=tsfm) if test else None
    return dataset_train, dataset_test

# model save and load

def add_softmax(model):
    if not isinstance(model[-1], nn.Softmax):
        model.add_module(str(len(model)), nn.Softmax(dim=1))
    return model

def find_latest_model(folder: str, prefix: str) -> tuple[str, int]:
    import os
    files = os.listdir(folder)
    files = [f for f in files if os.path.isfile(folder+"/"+f) and f.startswith(prefix) and f.endswith('.pt')]
    latest = 0
    for f in files:
        try:
            t = int(f[len(prefix):-3])
            if t > latest:
                latest = t
        except:
            pass
    if latest > 0:
        return "{}/{}{}.pt".format(folder, prefix, latest), latest
    return None, 0

def save_model_state(model, path: str):
    torch.save(model.state_dict(), path)

def load_model_state(model, path: str):
    model.load_state_dict(torch.load(path))
    return model

# train and test

def trainbatch(model, data, target, optimizer, loss_fn):
    # forward
    output = model(data)
    # loss
    loss = loss_fn(output, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model, dataset, batch_size: int=32, epochs: int=10, shuffle: bool=True,
          optimizer: torch.optim.Optimizer = None, loss_fn: nn.Module = None,
          *, n: int=None, show_interval: float = 60, device: str = 'cpu'):
    if device != 'cpu':
        assert torch.cuda.is_available(), 'CUDA is not available'
    model.to(device)
    model.train()
    # dataloader
    if n is None:
        n = len(dataset)
    if device == 'cpu':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # pin_memory_device is available for PyTorch >= 1.12
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        #                                          pin_memory=True, pin_memory_device=device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    # optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss function
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    # train
    t0 = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        running_loss = 0.0
        total = 0
        t1 = time.time()
        t2 = time.time()
        for batch in dataloader:
            data, target = batch
            if device != 'cpu':
                data = data.to(device)
                target = target.to(device)
            loss = trainbatch(model, data, target, optimizer, loss_fn)
            # print statistics
            total += len(target)
            running_loss += loss * len(target)
            if total > n:
                break
            t3 = time.time()
            t = t3 - t2
            if t >= show_interval:
                t2 = t3
                eta = t / total * (n - total)
                print('  Progress {:.1f}% ({}/{}). Loss: {:.8g}% Time: {:.2f}s ETA: {:.2f}s'.format(
                    total / n, total, n, running_loss/total, time.time() - t2, eta))
        running_loss /= total
        t = time.time() - t1
        eta = t * (epochs - epoch - 1)
        print('  Epoch {}: Loss: {:.8g} Time: {:.2f}s ETA: {:.2f}s'.format(epoch+1, running_loss, t, eta))
    print('Finished Training. Time: {:.2f}s'.format(time.time()-t0))


def testbatch(model, data, target):
    # forward
    output = model(data)
    # get prediction
    _, predicted = torch.max(output.data, 1)
    # count
    num = target.size(0)
    correct = (predicted == target).sum().item()
    return num, correct

def test(model, dataset, batch_size: int = 32, *, n: int = None, show_interval: float = 60, device: str = 'cpu'):
    if device != 'cpu':
        assert torch.cuda.is_available(), 'CUDA is not available'
    model.to(device)
    model.eval()
    # dataloader
    if n is None:
        n = len(dataset)
    if device == 'cpu':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    else:
        # pin_memory_device is available for PyTorch >= 1.12
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        #                                          pin_memory=True, pin_memory_device=device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    # test
    t0 = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        t1 = time.time()
        for batch in dataloader:
            # get data
            data, target = batch
            if device != 'cpu':
                data = data.to(device)
                target = target.to(device)
            m, c = testbatch(model, data, target)
            total += m
            correct += c
            if total > n:
                break
            t2 = time.time()
            t = t2 - t1
            if t >= show_interval:
                t1 = t2
                eta = t / c * (n - total)
                print('  Progress {:.1f}% ({}/{}). Accuracy: {:.2f}% Time: {:.2f}s ETA: {:.2f}s'.format(
                    total / n, total, n, 100 * correct / total, t, eta))
    print('Accuracy: {:.2f}% ({}/{}) Time: {:.2f}s'.format(
        100 * correct / total, correct, total, time.time()-t0))
    return correct / total


def process(model, trainset, testset, batch_size: int, epochs: int,
            optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
            dump_interval: int, dump_dir: str='', dump_prefix: str='',
            epoch_start = 0, accuracy_threshold = 0.0,
            device: str = 'cpu'):
    filename = 'NO FILE'
    if device != 'cpu':
        assert torch.cuda.is_available(), 'CUDA is not available'
    model.to(device)
    model.eval()
    # dataloader
    if device == 'cpu':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    # train, dump and test
    best_accuracy = accuracy_threshold
    best_loss = float('inf')
    t0 = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        # train
        running_loss = 0.0
        t1 = time.time()
        for data, target in trainloader:
            if device != 'cpu':
                data, target = data.to(device), target.to(device)
            loss = trainbatch(model, data, target, optimizer, loss_fn)
            running_loss += loss * len(target)
        running_loss /= len(trainset)
        if best_loss == float('inf'):
            best_loss = running_loss
        t2 = time.time()
        eta = (t2 - t0) / (epoch + 1) * (epochs - epoch - 1)
        print('  Epoch {}: Loss: {:.8g} Time: {:.2f}s ETA: {:.2f}s'.format(
            epoch+1, running_loss, t2 - t1, eta))
        # test and dump
        if epoch % dump_interval == dump_interval - 1:
            total = 0
            correct = 0
            t1 = time.time()
            with torch.no_grad():
                for data, target in testloader:
                    if device != 'cpu':
                        data, target = data.to(device), target.to(device)
                    n, c = testbatch(model, data, target)
                    total += n
                    correct += c
            accuracy = correct / total
            t2 = time.time()
            print('  Accuracy: {:.2f}% ({}/{}) Time: {:.2f}s'.format(
                100 * accuracy, correct, total, t2 - t1))
            if accuracy > best_accuracy or (accuracy == best_accuracy and running_loss < best_loss):
                best_accuracy = accuracy
                best_loss = running_loss
                if dump_dir == '' or dump_dir == '.':
                    filename = '{}{}.pt'.format(dump_prefix, epoch_start+epoch+1)
                else:
                    filename = '{}/{}{}.pt'.format(dump_dir, dump_prefix, epoch_start+epoch+1)
                print('  Dumping model to {}'.format(filename))
                save_model_state(model, filename)
    print('Finished Training. Time: {:.2f}s Best accuracy: {:.2f}%, loss: {:.2f} at file {}'.format(
        time.time()-t0, 100 * best_accuracy, best_loss, filename))
