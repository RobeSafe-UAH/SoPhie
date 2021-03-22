import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, dim_list, activation='relu', batch_norm=True, dropout=0):
        super(MLP, self).__init__()
        self.module = self.make_mlp(dim_list, activation, batch_norm, dropout)

    def make_mlp(self, dim_list, activation='relu', batch_norm=True, dropout=0):
        layers = []
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, data_input):
        return self.module(data_input)