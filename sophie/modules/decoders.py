import torch
from torch import nn
import torch.nn.functional as F
import pdb

from sophie.modules.layers import MLP, TrajConf

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
            traj_rel (20, b, 2)
            state_tuple: h and c
                h : c : (1, b, self.h_dim)
        """
        npeds = traj_abs.size(1)
        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(npeds, -1)))) # bx16
        decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim) # 1x batchx 16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) # output (1, b, 32)
            rel_pos = self.hidden2pos(self.ln2(output.contiguous().view(-1, self.h_dim))) # (b, 2)
            traj_rel = torch.roll(traj_rel, -1, dims=(0))
            traj_rel[-1] = rel_pos

            decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(npeds, -1))))
            decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(npeds,-1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel



class TemporalDecoderLSTMConf(nn.Module):

    def __init__(self, seq_len=30, h_dim=64, embedding_dim=32, modes=3, obs_len=20):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.modes = modes
        self.obs_len = 20
        self.traj_emb_dim = 2*self.obs_len*self.modes

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        # 2*20*3 = 120
        self.spatial_embedding = nn.Linear(self.traj_emb_dim, self.embedding_dim) # 20 obs * 2 points
        self.ln1 = nn.LayerNorm(self.traj_emb_dim)
        self.hidden2pos = nn.Linear(self.h_dim, 2*self.modes + self.modes) # (h_dim, 2*3 + 3)
        self.trajconf = TrajConf()
        self.ln2 = nn.LayerNorm(self.h_dim)

    def forward(self, traj_abs, traj_rel, state_tuple):
        """
        traj_abs (20, b, 2)
        """
        t, _, p = traj_rel.shape
        npeds = traj_abs.size(1)
        pred_traj_fake_rel = []
        # (20,3,2)
        traj_rel = torch.repeat_interleave(traj_rel.unsqueeze(0), 3, dim=0) # m, t, b, 2
        traj_rel = traj_rel.permute(2,0,1,3) # (b, m, t, 2)
        decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(npeds, -1)))) # bx16
        #(b, 120)
        decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim) # 1x batchx 16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            # (3,9) -> (b, 9)
            rel_pos = self.hidden2pos(self.ln2(output.contiguous().view(-1, self.h_dim))) # (h_dim, 2*3 + 3)
            traj_rel = torch.roll(traj_rel, -1, dims=(1))
            traj_rel = traj_rel.view(-1, ) # 3, 20, 3, 2
            traj_rel[-1] = rel_pos

            decoder_input = F.leaky_relu(self.spatial_embedding(self.ln1(traj_rel.contiguous().view(npeds, -1))))
            decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(npeds,-1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

"""
self.regressor = nn.Linear(dim_hidden, dim_output)
self.mode_confidences = nn.Linear(dim_hidden, 1)

coords = self.regressor(XX).reshape(-1, self.num_outputs, self.pred_len, 2) # (b, m, t, 2)
confidences = torch.squeeze(self.mode_confidences(XX), -1) # (b, m)
confidences = torch.softmax(confidences, dim=1)
"""

class BaseDecoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture.
    Defined in :numref:`sec_encoder-decoder`"""
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError