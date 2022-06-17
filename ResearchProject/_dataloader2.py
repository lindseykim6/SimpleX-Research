
import torch
from torch.utils.data import *
from matplotlib import pyplot as plt
import numpy as np


def sdf(x, y, r = 1):
    return torch.sqrt(x * x + y * y) - r


def sdf_circle(x, y, r = 1):
    return torch.sqrt(x * x + y * y) - r

def sdf_double_circle(x, y, h1, k1, r1, h2, k2, r2):
    d1 = torch.sqrt((x - h1) * (x - h1) + (y - k1) * (y - k1)) - r1
    d2 = torch.sqrt((x - h2) * (x - h2) + (y - k2) * (y - k2)) - r2
    tensor = torch.cat((d1, d2), 1)
    min, indices = torch.min(tensor, 1)
    d = min.reshape(len(min), 1)
    return d


def sdf_line(x, y, a, b, c):
    # ax + by + c = 0
    x=(a * x + b * y + c) / (np.sqrt(a * a + b * b))
    return x

def min_distance(x0, y0, x1, y1, x2, y2):
    px = x2 - x1
    py = y2 - y1
    norm = px * px + py * py
    u = ((x0 - x1) * px + (y0 - y1) * py) / float(norm)

    one = torch.ones_like(u)
    zero = torch.zeros_like(u)

    u = torch.where(u > 1, one, u)
    u = torch.where(u < 0, zero, u)

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x0
    dy = y - y0

    dist = torch.sqrt(dx*dx + dy*dy)
    print(dist)
    return dist


def sdf_line2(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4):
        ax = x2 - x1
        ay = y2 - y1
        bx = x4 - x3
        by = y4 - y3
        d1 = torch.abs((ax * (y1 - y0)) - (ay * (x1 - x0))) / np.sqrt(ax * ax + ay * ay)
        d2 = torch.abs((bx * (y3 - y0)) - (by * (x3 - x0))) / np.sqrt(bx * bx + by * by)
        d3 = torch.cat((d1, d2), 1)
        d4, indices = torch.min(d3, 1)
        d5 = d4.reshape(len(d4), 1)
        return d5


def sdf_triangle(x0,y0, x1, y1,x2, y2, x3, y3):
    # ax=x2-x1
    # ay=y2-y1
    # bx=x3-x2
    # by=y3-y2
    # cx=x3-x1
    # cy=y3-y1
    # d1=torch.abs((ax * (y1 - y0))-(ay * (x1 - x0)))/np.sqrt(ax * ax + ay * ay)
    # d2=torch.abs((bx * (y2 - y0)) - (by * (x2 - x0))) / np.sqrt(bx * bx + by * by)
    # d3=torch.abs((cx * (y3 - y0)) - (cy * (x3 - x0))) / np.sqrt(cx * cx + cy * cy)

    d1=min_distance(x0,y0,x1,y1,x2,y2)
    d2 = min_distance(x0, y0, x1, y1, x3, y3)
    d3 = min_distance(x0, y0, x2, y2, x3, y3)
    d4= torch.cat((d1,d2,d3),1)
    d5, indices=torch.min(d4,1)
    d6=d5.reshape(len(d5), 1)

    return d6

def sdf_square(x0,y0, x1, y1,x2, y2, x3, y3, x4, y4):
    # ax=x2-x1
    # ay=y2-y1
    # bx=x3-x2
    # by=y3-y2
    # cx=x4-x3
    # cy=y4-y3
    # dx=x4-x1
    # dy=y4-y1
    # d1=torch.abs((ax * (y1 - y0))-(ay * (x1 - x0)))/np.sqrt(ax * ax + ay * ay)
    # d2=torch.abs((bx * (y2 - y0)) - (by * (x2 - x0))) / np.sqrt(bx * bx + by * by)
    # d3=torch.abs((cx * (y3 - y0)) - (cy * (x3 - x0))) / np.sqrt(cx * cx + cy * cy)
    # d4 = torch.abs((dx * (y4 - y0)) - (dy * (x4 - x0))) / np.sqrt(dx * dx + dy * dy)
    d1 = min_distance(x0, y0, x1, y1, x2, y2)
    d2 = min_distance(x0, y0, x2, y2, x3, y3)
    d3 = min_distance(x0, y0, x3, y3, x4, y4)
    d4 = min_distance(x0, y0, x1, y1, x4, y4)
    d5= torch.cat((d1,d2,d3, d4),1)
    d6, indices=torch.min(d5,1)
    d7=d6.reshape(len(d5), 1)
    print(d7)

    return d7

def generate_2dim(num, r=1, randomness=0.8):
    pi = torch.acos(torch.zeros(1)).item() * 2
    phi = torch.rand([num, 1], out=None) * 2 * pi
    x = torch.cos(phi)
    y = torch.sin(phi)

    n_x = (x+torch.randn([num, 1]) * randomness) * r
    n_y = (y+torch.randn([num, 1]) * randomness) * r
    data = torch.cat((n_x, n_y), 1)

    n_x.requires_grad_(True)
    n_y.requires_grad_(True)
    #d = sdf_line (n_x, n_y, 2,0,2)
    #d=min_distance(n_x,n_y,0,0,1,1)
    #d = sdf_triangle(n_x, n_y,0,0, 0.5,0,-1,1)
    #d = sdf_square(n_x, n_y, 0, 1, 1, 1, 1, 0, 0, 0)
    #d = sdf_line2(n_x, n_y, 0, 2, 0,0, 2,2, 2,0)
    d = sdf_circle(n_x, n_y, r=1)
    d=sdf_double_circle(n_x, n_y, 0,0, 1, 1, 1, 1)

    d.sum().backward()
    x2 = n_x - d * n_x.grad
    y2 = n_y - d * n_y.grad


    label = torch.cat((x2, y2), 1)

    # label = data / torch.sqrt(torch.sum(data * data, dim=1)).expand(2, -1).permute(1, 0) * r
    return data.view(num, 1, 2), label.view(num, 1, 2)


# 51200
def get_data_loader_sphere(num=10, split=0.8, use_cuda=False):
    d, l = generate_2dim(num)
    d = d.detach()
    l = l.detach()
    #plot(d, l)
    idx = int(num * split)
    d_tr = (d[0:idx, :, :].cuda() if use_cuda else d[0:idx, :, :])
    d_val = (d[idx:len(d), :, :].cuda() if use_cuda else d[idx:len(d), :, :])
    l_tr = (l[0:idx, :, :].cuda() if use_cuda else l[0:idx, :, :])
    l_val = (l[idx:len(d), :, :].cuda() if use_cuda else l[idx:len(d), :, :])

    train_ds = TensorDataset(d_tr, l_tr)
    val_ds = TensorDataset(d_val, l_val)

    print("TRAINING")
    print(train_ds.tensors)
    print(val_ds.tensors)
    return train_ds, val_ds


def plot(d, l):
    plt.scatter(d[:, 0, 0], d[:, 0, 1], c='r')
    plt.scatter(l[:, 0, 0], l[:, 0, 1], c='b')
    plt.show()


if __name__ == '__main__':
    d, l = generate_2dim(500)
    d = d.detach().numpy()
    l = l.detach().numpy()

    plot(d, l)

