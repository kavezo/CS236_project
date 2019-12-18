# Copyright (c) 2018 Rui Shu
import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

x_dim = (1, 80, 157) # hardcode...
layers_conf = [(64, (80, 1), (1, 1), None), # also hardcode...
                (128, (3,1), (2,1), 'same'),
                (256, (3,1), (2,1), 'same')]
conv_size = None

class ConvEncoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        global conv_size
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim

        in_channels = [1] + [opt[0] for opt in layers_conf]
        layers = []
        curr_size = x_dim
        for i in range(len(layers_conf)):
            out_channels, kernel_size, stride, padding_type = layers_conf[i]
            padding = (int(kernel_size[0]/2), int(kernel_size[1]/2)) if padding_type else (0,0)
            layers.append(torch.nn.Conv2d(in_channels=in_channels[i],
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.BatchNorm2d(out_channels))

            curr_size = (out_channels,
                         int((curr_size[1]+(2*padding[0])-(kernel_size[0]-1)-1)/stride[0])+1,
                         int((curr_size[2]+(2*padding[1])-(kernel_size[1]-1)-1)/stride[1])+1)

        conv_size = curr_size

        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(int(np.prod(curr_size)), 2*z_dim))

        self.net = torch.nn.Sequential(*layers)

    def encode(self, x):
        h = self.net(x)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class ConvDecoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        global conv_size
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim

        out_channels = [1] + [opt[0] for opt in layers_conf[:-1]]
        layers = []

        layers.append(torch.nn.Linear(z_dim, int(np.prod(conv_size))))
        layers.append(Reshape(conv_size))

        for i in reversed(range(len(layers_conf))):
            in_channels, kernel_size, stride, padding_type = layers_conf[i]
            padding = (int(kernel_size[0] / 2), int(kernel_size[1] / 2)) if padding_type else (0, 0)
            layers.append(torch.nn.ConvTranspose2d(in_channels=in_channels,
                                    out_channels=out_channels[i],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.BatchNorm2d(out_channels[i]))

        self.net = torch.nn.Sequential(*layers)

    def decode(self, z):
        return self.net(z)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)