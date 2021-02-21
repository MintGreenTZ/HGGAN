# modify from https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_pytorch.py

import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

from pose_loader import Pose_Loader

from display import show_hand_mano

import datetime
import dateutil
import dateutil.tz

ho3d = Pose_Loader()

mb_size = 64
Z_dim = 100
X_dim = ho3d.get_true_pose_shape()[0]
y_dim = ho3d.get_condition_shape()[0]
h_dim = 128
cnt = 0
lr = 1e-3
print("Z_dim " + str(Z_dim))
print("X_dim " + str(X_dim))
print("y_dim " + str(y_dim))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[Z_dim + y_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def G(z, c):
    inputs = torch.cat([z, c], 1)
    h = nn.relu(inputs @ Wzh + bzh.repeat(inputs.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = Variable(torch.zeros(1), requires_grad=True)


def D(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(inputs @ Wxh + bxh.repeat(inputs.size(0), 1))
    y = nn.sigmoid(h @ Why + bhy.repeat(h.size(0), 1))
    return y


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
params = G_params + D_params


""" ===================== TRAINING ======================== """


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


G_solver = optim.Adam(G_params, lr=1e-3)
D_solver = optim.Adam(D_params, lr=1e-3)

ones_label = Variable(torch.ones(mb_size, 1))
zeros_label = Variable(torch.zeros(mb_size, 1))

def ensure_dir(aim_dir):
    if not os.path.exists(aim_dir):
        os.makedirs(aim_dir)

def align_number(number, bit = 6):
    out = str(number)
    while len(out) < bit:
        out = "0" + out
    return out

output_dir = "./output"
ensure_dir(output_dir)
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
save_dir = os.path.join(output_dir, timestamp)
ensure_dir(save_dir)
for epoch in range(1000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim))
    X, c = ho3d.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))
    c = Variable(torch.from_numpy(c.astype('float32')))

    # Dicriminator forward-loss-backward-update
    print(z.shape)
    print(c.shape)
    G_sample = G(z, c)
    D_real = D(X, c)
    D_fake = D(G_sample, c)

    D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
    D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim))
    G_sample = G(z, c)
    D_fake = D(G_sample, c)

    G_loss = nn.binary_cross_entropy(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Print and plot every now and then
    if epoch % 20 == 0:
        print('Epoch-{}; D_loss: {}; G_loss: {}'.format(epoch, D_loss.data.numpy(), G_loss.data.numpy()))

        cur_save_dir = os.path.join(save_dir, align_number(epoch))
        ensure_dir(cur_save_dir)

        print(c.shape)
        print(G_sample.shape)

        N = G_sample.shape[0]
        for i in range(N):
            obj_idx, obj_cloud = ho3d.get_obj(int(c[i][0]))
            save_file_name = os.path.join(cur_save_dir, "output_" + align_number(i, 3) + ".png")
            # show_hand_mano(np.zeros(48),np.zeros(3),np.zeros(10), obj_cloud, save_file_name)
            sample = G_sample[i].detach().numpy()
            print(type(obj_cloud))
            print(obj_cloud)
            print(obj_cloud.shape)
            # obj_cloud = np.random.randn(262146, 3)
            show_hand_mano(sample[0:48], sample[48:51], sample[51:61], obj_cloud, save_file_name)