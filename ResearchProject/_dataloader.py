import torch
from torch.utils.data import *
from matplotlib import pyplot as plt
import numpy as np

def generate_2dim(num, r = 1, randomness = 0.2):
    pi = torch.acos(torch.zeros(1)).item() * 2 
    phi = torch.rand([num, 1], out=None) * 2 * pi
    x = torch.cos(phi)
    y = torch.sin(phi)
    print(x)
    n_x = (x + torch.randn([num, 1]) * randomness) * r
    print(n_x)
    n_y = (y + torch.randn([num, 1]) * randomness) * r
    data = torch.cat((n_x, n_y), 1)
    label = data / torch.sqrt(torch.sum(data * data, dim = 1)).expand(2, -1).permute(1, 0) * r
    return data.view(num, 1, 2), label.view(num, 1, 2)

def get_data_loader_sphere(num = 51200, split=0.8, use_cuda=False):
    d, l= generate_2dim(num)
    # plot(d, l)
    idx = int(num * split)
    d_tr = (d[0:idx, :, :].cuda() if use_cuda else d[0:idx, :, :])
    d_val = (d[idx:len(d), :, :].cuda() if use_cuda else d[idx:len(d), :, :])
    l_tr = (l[0:idx, :, :].cuda() if use_cuda else l[0:idx, :, :])
    l_val = (l[idx:len(d), :, :].cuda() if use_cuda else l[idx:len(d), :, :])
    print(d_tr)
    print(l_tr)
    train_ds = TensorDataset(d_tr, l_tr)
    val_ds = TensorDataset(d_val, l_val)
    return train_ds, val_ds

def plot(d, l):
    plt.scatter(d[:,0,0], d[:,0,1], c = 'r')
    plt.scatter(l[:,0,0], l[:,0,1], c = 'b')
    plt.show()
    
if __name__ == '__main__':
    d, l= generate_2dim(100)
    training, value = get_data_loader_sphere()
    plot(d, l)


