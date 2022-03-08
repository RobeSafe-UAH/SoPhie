from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torchvision.transforms.functional as TF
from sophie.modules.attention import TransformerEncoder

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len=20, pred_len=30, mlp_dim=64, h_dim=32, embedding_dim=16, bottleneck_dim=32,
        noise_dim=8, n_agents=10, img_feature_size=(512,6,6), dropout=0
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.bottleneck_dim = bottleneck_dim
        self.noise_dim = noise_dim
        self.n_agents = n_agents

        ## Transformer encoder
        self.encoder = TransformerEncoder(
            2, h_dim, h_dim, h_dim, h_dim, h_dim,
            h_dim, h_dim*2, 8, 2, .5
        )

        self.ln1 = nn.LayerNorm(self.h_dim)

        ## Transformer decoder

        ## Trajectory regressor
        self.regressor_input = self.h_dim*self.obs_len
        self.regressor_output = self.pred_len*2
        self.regressor = nn.Linear(self.regressor_input, self.regressor_output)

    def forward(self, obs_traj_rel, start_end_seq, agent_idx=None):
        """
            n: objects in seq
            b: batch
            obs_traj: (20,n*b,2)
            obs_traj_rel: (20,n*b,2)
            start_end_seq: (b,2)
            agent_idx: (b, 1) -> index of agent in every sequence.
                None: trajectories for every object in the scene will be generated
                Not None: just trajectories for the agent in the scene will be generated
            -----------------------------------------------------------------------------
            pred_traj_fake_rel_batch:
                (30,n*b,2) -> if agent_idx is None
                (30,b,2)
        """

        ## Encode trajectory -> transformer encoder
        _, nb, _ = obs_traj_rel.shape
        x = []
        for start, end in start_end_seq.data:
            y = obs_traj_rel[:,start:end,:].contiguous().permute(1,0,2) # (obs,n,2)
            n, t, p = y.shape
            y = y.contiguous().view(1, -1, p) # (1, obs*n, 2)
            y = self.ln1(self.encoder(
                y,
                None
            )) # (1, obs*n, h_dim)
            y = y.view(n,t,self.h_dim) # (n, obs, h_dim)
            x.append(y)
        x = torch.cat(x, 0) # (n*b, obs, h_dim)

        ## Decode trajectory
        x = self.regressor(x.view(nb,-1)).view(nb,self.pred_len,-1)
        return x.permute(1,0,2)

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, mlp_dim=64, h_dim=64):
        super(TrajectoryDiscriminator, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim

        self.encoder = None
        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims)

    def forward(self, traj, traj_rel):

        final_h = self.encoder(traj_rel)
        scores = self.real_classifier(final_h)
        return scores