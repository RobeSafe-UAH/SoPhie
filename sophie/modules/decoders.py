import torch
from torch import nn
import torch.nn.functional as F

from sophie.modules.layers import MLP

class DecoderLSTM(nn.Module):
 
    def __init__(self, seq_len=30, h_dim=64, embedding_dim=16):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.ln1 = nn.LayerNorm(2)
        self.hidden2pos = nn.Linear(self.h_dim, 2)
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

class TemporalDecoderLSTM(nn.Module):

    def __init__(self, seq_len=30, h_dim=64, embedding_dim=16):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(40, self.embedding_dim) # 20 obs * 2 points
        self.ln1 = nn.LayerNorm(40)
        self.hidden2pos = nn.Linear(self.h_dim, 2)
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