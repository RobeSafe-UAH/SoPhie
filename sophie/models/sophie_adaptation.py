from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torchvision.transforms.functional as TF
from sophie.modules.backbones import VisualExtractor

MAX_PEDS = 32

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)

def get_noise(shape):
    return torch.randn(*shape).cuda()

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.
    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # 1x80x10x32

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3) # 1x10x80x32

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3]) # 

def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.
    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.
    Defined in :numref:`sec_utils`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return F.softmax(X,dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)

class Encoder(nn.Module):

    def __init__(self, embedding_dim=16, h_dim=64):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

    def init_hidden(self, batch):
        h = torch.zeros(1,batch, self.h_dim).cuda()
        c = torch.zeros(1,batch, self.h_dim).cuda()
        return h, c

    def forward(self, obs_traj):

        npeds = obs_traj.size(1)

        obs_traj_embedding = F.leaky_relu(self.spatial_embedding(obs_traj.contiguous().view(-1, 2)))
        obs_traj_embedding = obs_traj_embedding.view(-1, npeds, self.embedding_dim)
        state = self.init_hidden(npeds)
        output, state = self.encoder(obs_traj_embedding, state)
        final_h = state[0]
        final_h = final_h.view(npeds, self.h_dim)
        return final_h

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

class DecoderTemporal(nn.Module):

    def __init__(self, seq_len=30, h_dim=64, embedding_dim=16):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(40, self.embedding_dim) # 2 (x,y) | 20 num_obs
        self.hidden2pos = nn.Linear(self.h_dim, 2)
        self.output_activation = nn.Sigmoid()

    def forward(self, traj_abs, traj_rel, state_tuple):
        """
        traj_abs (20, b, 2)
        """
        npeds = traj_abs.size(1)
        pred_traj_fake_rel = []
        decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel.contiguous().view(npeds, -1))) # bx16
        decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim) # 1x batchx 16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) #
            rel_pos = F.leaky_relu(self.output_activation(self.hidden2pos(output.contiguous().view(-1, self.h_dim))))# + last_pos_rel # 32 -> 2
            traj_rel = torch.roll(traj_rel, -1, dims=(0))
            traj_rel[-1] = rel_pos

            decoder_input = F.leaky_relu(self.spatial_embedding(traj_rel.contiguous().view(npeds, -1)))
            decoder_input = decoder_input.contiguous().view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(npeds,-1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

class DotProductAttention(nn.Module):
    """
    Scaled dot product attention
        a(q,k) = q*k/sqrt(d)
        Computationally more effient than Additive. This attention requires "d" to keep the variance
        in q and k. Means its not modified.
    """

    def __init__(self, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, valid_lens=None):
        """
        Shape of `queries`: (`batch_size`, no. of queries, `d`)
        Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
        Shape of `values`: (`batch_size`, no. of key-value pairs, value
            dimension)
        Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
        Multi-head attention combines knowledge of the same attention pooling via different 
        representation subspaces of queries, keys, and values.
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        #TODO select attention mechanism: additive or dotproduct
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        Shape of `queries`, `keys`, or `values`:
            (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        Shape of `valid_lens`:
            (`batch_size`,) or (`batch_size`, no. of queries)
        After transposing, shape of output `queries`, `keys`, or `values`:
            (`batch_size` * `num_heads`, no. of queries or key-value pairs,
            `num_hiddens` / `num_heads`)
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class PositionalEncoding(nn.Module):
    """
    Positional encoding (absolute)
        input: X
        output: X + P
        P: positional embedding matrix
    """

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1,1) / torch.pow(
            10000, torch.arange(0,num_hiddens,2,dtype=torch.float32) / num_hiddens
        )
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len=20, pred_len=30, mlp_dim=64, h_dim=32, embedding_dim=16, bottleneck_dim=32,
        noise_dim=8, n_agents=10, img_feature_size=(512,6,6), dropout=0 # TODO modificar img_feature_size para que se modifique con el modelo
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
        self.img_feats = img_feature_size[0]*img_feature_size[1]*img_feature_size[2]

        vgg_config = {
            "vgg_type": 19,
            "batch_norm": False,
            "pretrained": True,
            "features": True
        }
        self.visual_feature_extractor = VisualExtractor(
            "vgg19",
            vgg_config
        )

        for param in self.visual_feature_extractor.parameters():
            param.requires_grad = False


        self.encoder = Encoder(h_dim=self.h_dim)
        self.lne = nn.LayerNorm(self.h_dim)
        self.sattn = MultiHeadAttention(
            key_size=self.h_dim, query_size=self.h_dim, value_size=self.h_dim, num_hiddens=self.h_dim, num_heads=4, dropout=dropout
        )
        self.pos_encoding = PositionalEncoding(self.img_feats, dropout) # image
        self.pattn = MultiHeadAttention(
            key_size=self.img_feats, query_size=self.h_dim, value_size=self.img_feats, num_hiddens=self.h_dim, num_heads=4, dropout=dropout
        )
        self.decoder = Decoder(h_dim=self.h_dim)

        # mlp_decoder_context_dims = [self.h_dim*3, self.mlp_dim, self.h_dim - self.noise_dim]
        mlp_context_input = self.h_dim*2
        self.lnc = nn.LayerNorm(mlp_context_input)
        mlp_decoder_context_dims = [mlp_context_input, self.mlp_dim, self.h_dim - self.noise_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims) # [96, 64, 44]

    def add_noise(self, _input):
        npeds = _input.size(0)
        noise_shape = (self.noise_dim,)
        z_decoder = get_noise(noise_shape)
        vec = z_decoder.view(1, -1).repeat(npeds, 1)
        return torch.cat((_input, vec), dim=1)

    def calculate_patch(self, img_tensor): #TODO remove magic number
        # batch x 512 x 18 x 18
        stride = int(img_tensor.size(2)/3)
        batch = img_tensor.size(0)
        patch = []
        for i in range(3):
            for j in range(3):
                patch.append(img_tensor[:,:,i*stride:(i+1)*stride, (j)*stride:(j+1)*stride].contiguous().view(batch,-1))
        return torch.stack(patch, dim=1) # batch x 9 x (512*6*6)=18432

    def forward(self, obs_traj, obs_traj_rel, frames, agent_idx=None):

        # visual_features = self.visual_feature_extractor(frames) # batch x 512 x 18 x 18
        # visual_patch = self.calculate_patch(visual_features) # 8x9x(512*6*6)
        # visual_patch_enc = self.pos_encoding(visual_patch)
        # visual_patch_enc = visual_patch_enc.contiguous().view(
        #     1, visual_patch_enc.size(0)*visual_patch_enc.size(1), visual_patch_enc.size(2)
        # )
        
        npeds = obs_traj_rel.size(1)
        final_encoder_h = self.encoder(obs_traj_rel) # batchx32
        # final_encoder_h = final_encoder_h.contiguous().view(batch, -1, self.h_dim) # 8x10x32
        final_encoder_h = torch.unsqueeze(final_encoder_h, 0) #  1xbatchx32

        # queries -> indican la forma del tensor de salida (primer argumento)
        final_encoder_h = self.lne(final_encoder_h)
        attn_s = self.sattn(final_encoder_h, final_encoder_h, final_encoder_h, None) # 8x10x32 # multi head self attention
        # attn_p = self.pattn(final_encoder_h, visual_patch_enc, visual_patch_enc, None) # 8x10x32
        
        mlp_decoder_context_input = torch.cat(
            [
                final_encoder_h.contiguous().view(-1, 
                self.h_dim), attn_s.contiguous().view(-1, self.h_dim), 
                # attn_p.contiguous().view(-1, self.h_dim)
            ],
            dim=1
        ) # 80 x (32*3)
        if agent_idx is not None:
            mlp_decoder_context_input = mlp_decoder_context_input[agent_idx,:]

        noise_input = self.mlp_decoder_context(self.lnc(mlp_decoder_context_input)) # 80x24
        decoder_h = self.add_noise(noise_input) # 80x32
        decoder_h = torch.unsqueeze(decoder_h, 0) # 1x80x32

        decoder_c = torch.zeros(tuple(decoder_h.shape)).cuda() # 1x80x32
        state_tuple = (decoder_h, decoder_c)

        if agent_idx is not None: # for single agent prediction
            last_pos = obs_traj[-1, agent_idx, :]
            last_pos_rel = obs_traj_rel[-1, agent_idx, :]
        else:
            last_pos = obs_traj[-1, :, :]
            last_pos_rel = obs_traj_rel[-1, :, :]

        pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
        return pred_traj_fake_rel

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, mlp_dim=64, h_dim=64):
        super(TrajectoryDiscriminator, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim

        self.encoder = Encoder()
        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims)

    def forward(self, traj, traj_rel):

        final_h = self.encoder(traj_rel)
        scores = torch.sigmoid(self.real_classifier(final_h))
        return scores