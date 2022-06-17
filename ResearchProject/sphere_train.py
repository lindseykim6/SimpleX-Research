from _dataloader2 import *
from _training import *
from _networks import *
from _iterative_proj import *

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

MODEL_DIR = "models/"
NAME = "sphere"
NUM_PARTICLES = 1
DIMENSION = 2
NUM_ITER = 5
NN_LAYERS = [64, 32, 1]

DATASET_SAMPLE_NUM = 4096
DATASET_SPLIT = 0.8
DATASET_SPLIT = 0.8

USE_CUDA = False

def training_main():
    train_opts = {
        "num_epochs": 70,
        "lr": 1e-3,
        'lr_step': 22,
        'lr_gamma': 0.8,
        "batch_size": 128,
        "loss": 'l1',
        "weight_decay": 0 
    }

    train_ds, val_ds = get_data_loader_sphere(num=DATASET_SAMPLE_NUM, split=DATASET_SPLIT, use_cuda=USE_CUDA)
    print("train")
    print(train_ds)
    print("val")
    print(val_ds)
    func_net = Simple_NN(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=NN_LAYERS)
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, func=func_net, num_iter=NUM_ITER)
    if USE_CUDA:
        proj_model = proj_model.cuda()

    exp_dir = MODEL_DIR + NAME + "/"
    print("========================\n The Projection Module: ")
    print(proj_model)
    print("========================\n Training: ")
    train(proj_model, train_ds, val_ds, train_opts=train_opts, exp_dir=exp_dir)

if __name__ == '__main__':
    training_main()
