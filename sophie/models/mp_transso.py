from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torchvision.transforms.functional as TF
from sophie.modules.backbones import VisualExtractor
from sophie.modules.attention import TransformerEncoder

MAX_PEDS = 32

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)

def get_noise(shape):
    return torch.randn(*shape).cuda()

class Decoder(nn.Module):

    def __init__(self, seq_len=30, h_dim=64, embedding_dim=16):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.ln1 = nn.LayerNorm(2)
        self.hidden2pos = nn.Linear(self.h_dim, 2)
        # self.hidden2pos = make_mlp([self.h_dim, 128, 64, 2])
        self.ln2 = nn.LayerNorm(self.h_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, last_pos, last_pos_rel, state_tuple):
        npeds = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(last_pos_rel))) # 16
        decoder_input = decoder_input.view(1, npeds, self.embedding_dim) # 1x batchx 16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) #
            rel_pos = self.output_activation(self.hidden2pos(self.ln2(output.view(-1, self.h_dim))))# + last_pos_rel # 32 -> 2
            curr_pos = rel_pos + last_pos
            embedding_input = rel_pos

            decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(embedding_input)))
            decoder_input = decoder_input.view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(npeds,-1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

class TemporalDecoder(nn.Module):

    def __init__(self, seq_len=30, h_dim=64, embedding_dim=16):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(40, self.embedding_dim) # 20 obs * 2 points
        self.ln1 = nn.LayerNorm(40)
        self.hidden2pos = nn.Linear(self.h_dim, 2)
        # self.hidden2pos = make_mlp([self.h_dim, 128, 64, 2])
        self.ln2 = nn.LayerNorm(self.h_dim)

    def forward(self, traj_abs, traj_rel, state_tuple):
        """
        traj_abs (20, b, 2)
        """
        npeds = traj_abs.size(1)
        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(npeds, -1)))) # bx16
        decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim) # 1x batchx 16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) 
            rel_pos = self.hidden2pos(self.ln2(output.contiguous().view(-1, self.h_dim)))
            traj_rel = torch.roll(traj_rel, -1, dims=(0))
            traj_rel[-1] = rel_pos

            decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(npeds, -1))))
            decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(npeds,-1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel


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

        self.encoder = TransformerEncoder()

        mlp_context_input = self.h_dim # concat of social context and trajectories embedding
        self.lnc = nn.LayerNorm(mlp_context_input)
        mlp_decoder_context_dims = [mlp_context_input, self.mlp_dim, self.h_dim - self.noise_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)


        self.decoder = TemporalDecoder(h_dim=self.h_dim)

    def add_noise(self, _input):
        npeds = _input.size(0)
        noise_shape = (self.noise_dim,)
        z_decoder = get_noise(noise_shape)
        vec = z_decoder.view(1, -1).repeat(npeds, 1)
        return torch.cat((_input, vec), dim=1)

    def forward(self, obs_traj, obs_traj_rel, start_end_seq, agent_idx=None):
        """
            n: number of objects in all the scenes of the batch
            b: batch
            obs_traj: (20,n,2)
            obs_traj_rel: (20,n,2)
            start_end_seq: (b,2)
            agent_idx: (b, 1) -> index of agent in every sequence.
                None: trajectories for every object in the scene will be generated
                Not None: just trajectories for the agent in the scene will be generated
            -----------------------------------------------------------------------------
            pred_traj_fake_rel_batch:
                (30,n,2) -> if agent_idx is None
                (30,b,2)
        """
        
        ## Encode trajectory -> transformer encoder
        final_encoder_h = []
        for start, end in start_end_seq.data:
            encoder_h = self.encoder(obs_traj_rel[:,start:end,:]) # batchx32
            final_encoder_h.append(encoder_h)
        final_encoder_h = torch.cat(final_encoder_h, 1)

        final_encoder_h = self.lne(final_encoder_h)
        
        if agent_idx is not None:
            final_encoder_h = final_encoder_h[agent_idx,:]

        ## add noise to decoder input
        noise_input = self.mlp_decoder_context(self.lnc(final_encoder_h)) # 80x24
        decoder_h = self.add_noise(noise_input) # 80x32
        decoder_h = torch.unsqueeze(decoder_h, 0) # 1x80x32

        decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda() # 1x80x32
        state_tuple = (decoder_h, decoder_c)

        # Get agent observations
        if agent_idx is not None: # for single agent prediction
            last_pos = obs_traj[:, agent_idx, :]
            last_pos_rel = obs_traj_rel[:, agent_idx, :]
        else:
            last_pos = obs_traj[-1, :, :]
            last_pos_rel = obs_traj_rel[-1, :, :]

        ## decode trajectories -> transformer decoder
        pred_traj_fake_rel_batch = []
        for start, end in start_end_seq.data:
            pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
            pred_traj_fake_rel_batch.append(pred_traj_fake_rel)
        pred_traj_fake_rel_batch = torch.cat(pred_traj_fake_rel_batch, 1)
        return pred_traj_fake_rel_batch

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