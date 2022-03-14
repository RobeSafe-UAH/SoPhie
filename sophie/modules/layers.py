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


class TrajConf(nn.Module):

    def __init__(self, emb_dim=32, modes=3, pred_len=30):
        super().__init__()
        
        self.emb_dim=emb_dim
        self.modes = modes # 3
        self.pred_len = pred_len # 30
        self.preds = self.modes*2*self.pred_len # 2 points * 30 preds = 60 * numbers of trajs = 180 + 3 (conf)

        self.logit = nn.Linear(
            self.emb_dim, out_features=self.preds + self.modes
        )

    def forward(self, x):
        """
            x: (?)
        """
        x = self.logit(x)
        b, _ = x.shape
        preds, conf = torch.split(x, self.preds, dim=1)
        # pred = pred.view(10, 3, 30, 2)
        preds = preds.view(b, self.modes, self.pred_len, 2)
        conf = torch.softmax(conf, dim=1)
        return preds, conf
        