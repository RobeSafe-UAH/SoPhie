#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 24 23:53:07 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import rand
import pdb

class get_gumbel_map(nn.Module):
    def __init__(self, grid_size):
        super(get_gumbel_map, self).__init__()

        x = torch.arange(0, grid_size * 2 + 1)
        x = x.unsqueeze(1)
        X = x.repeat(1, grid_size * 2 + 1)

        x1 = X - grid_size
        x2 = x1.T

        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)

        self.gumbel_map = torch.cat((x2, x1), 2).view(1, -1, 2)

    def forward(self, batch_size):
        gumbel_map = self.gumbel_map.repeat(batch_size, 1, 1).float()
        gumbel_map = gumbel_map + torch.rand_like(gumbel_map)
        return gumbel_map

class GumbelSoftmax(nn.Module):
    def __init__(self, hard=False, temp=None):
        super(GumbelSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False
        self.temp = temp

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return Variable(noise).cuda()
        else:
            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, alpha, temperature, eps=1e-10):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = len(alpha.size()) - 1
        gumble_samples_tensor = self.sample_gumbel_like(alpha.data)

        gumble_trick_log_prob_samples = alpha + gumble_samples_tensor
        gumble_log_temp = gumble_trick_log_prob_samples / temperature
        max_gumble, _ = gumble_log_temp.max(1)
        soft_samples_gumble = F.softmax(gumble_log_temp - max_gumble.unsqueeze(1), dim)
        soft_samples_gumble = torch.max(soft_samples_gumble, torch.ones_like(soft_samples_gumble).to(alpha) * eps)
        soft_samples = F.softmax(alpha, dim)
        return soft_samples_gumble, soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        soft_samples_gumble, soft_samples = self.gumbel_softmax_sample(logits, temperature)
        if hard:

            _, max_value_indexes = soft_samples_gumble.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)

            y = y_hard - soft_samples_gumble.data + soft_samples_gumble

        else:
            y = soft_samples_gumble
        return y, soft_samples_gumble, soft_samples

    def forward(self, alpha, temp=None, force_hard=False):
        if not temp:
            if self.temp:
                temp = self.temp
            else:
                temp = 1

        if self.training and not force_hard:

            return self.gumbel_softmax(alpha, temperature=temp, hard=False)
        else:

            return self.gumbel_softmax(alpha, temperature=temp, hard=True)

class GumbelSampler(nn.Module):

    def __init__(self,
                 temp=1,
                 grid_size_out=16,
                 scaling=0.5,
                 force_hard=True,
                 ):
        super(GumbelSampler, self).__init__()
        self.temp = temp
        self.grid_size_out = grid_size_out
        self.scaling = scaling
        self.gumbelsoftmax = GumbelSoftmax(temp=self.temp)
        self.gumbel_map = get_gumbel_map(grid_size=self.grid_size_out)
        self.force_hard = force_hard

    def forward(self, cnn_out):
        """
        :param cnn_out:
        :type cnn_out:
        :return:
            final_pos: Tensor with probability for each position
            final_pos_map: final_pos tensor reshaped
            y_softmax_gumbel: tensor with gumbel probabilities
            y_softmax: tensor with probabilites
        :rtype:
        """

        batch_size, c, hh, w = cnn_out["PosMap"].size()

        gumbel_map = self.gumbel_map(batch_size).to(cnn_out["PosMap"])
        y_scores = cnn_out["PosMap"].view(batch_size, -1)

        final_pos_map, y_softmax_gumbel, y_softmax = self.gumbelsoftmax(y_scores, force_hard=self.force_hard)

        final_pos = torch.sum(gumbel_map * final_pos_map.unsqueeze(2), 1).unsqueeze(0)

        final_pos_map = final_pos_map.view(batch_size, c, hh, w)
        y_softmax_gumbel = y_softmax_gumbel.view(batch_size, c, hh, w)
        y_softmax = y_softmax.view(batch_size, c, hh, w)
        final_pos = final_pos * self.scaling

        return final_pos, final_pos_map, y_softmax_gumbel, y_softmax, y_scores

def make_mlp(dim_list, activation_list, batch_norm=False, dropout=0):
    """
    Generates MLP network:
    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)
    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        activation = activation_list[index]
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))
        index += 1
    return nn.Sequential(*layers)

class UpConv_Blocks(nn.Module):
    def __init__(self, input_dim, output_dim, filter=4, padding=1, first_block=False, last_block=False,
                 batch_norm=False, non_lin="relu", dropout=0, skip_connection=False):
        super(UpConv_Blocks, self).__init__()
        self.Block = nn.Sequential()
        self.skip_connection = skip_connection
        self.first_block = first_block
        self.last_block = last_block
        if self.skip_connection and not self.first_block:
            ouput_dim_conv = input_dim
            input_dim *= 2
        else:
            ouput_dim_conv = output_dim

        self.Block.add_module("UpConv", nn.ConvTranspose2d(input_dim, output_dim, filter, 2, padding))
        if not last_block:
            if batch_norm:
                self.Block.add_module("BN_up", nn.BatchNorm2d(output_dim))
            if non_lin == "tanh":
                self.Block.add_module("NonLin_up", nn.Tanh())
            elif non_lin == "relu":
                self.Block.add_module("NonLin_up", nn.ReLU())
            elif non_lin == "leakyrelu":
                self.Block.add_module("NonLin_up", nn.LeakyReLU())
            if dropout > 0:
                self.Block.add_module("Drop_up", nn.Dropout2d(dropout))

    def forward(self, x, ):
        if self.skip_connection:
            x, skip_con_list = x
            if not self.first_block:
                x = torch.cat((x, skip_con_list.pop(-1)), -3)
        x = self.Block(x)

        if self.skip_connection and not self.last_block:
            x = [x, skip_con_list]
        return x

class Conv_Blocks(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size=3, batch_norm=False, non_lin="tanh", dropout=0.,
                 first_block=False, last_block=False, skip_connection=False):
        super(Conv_Blocks, self).__init__()
        self.skip_connection = skip_connection
        self.last_block = last_block
        self.first_block = first_block
        self.Block = nn.Sequential()
        self.Block.add_module("Conv_1", nn.Conv2d(input_dim, output_dim, filter_size, 1, 1))
        if batch_norm:
            self.Block.add_module("BN_1", nn.BatchNorm2d(output_dim))
        if non_lin == "tanh":
            self.Block.add_module("NonLin_1", nn.Tanh())
        elif non_lin == "relu":
            self.Block.add_module("NonLin_1", nn.ReLU())
        elif non_lin == "leakyrelu":
            self.Block.add_module("NonLin_1", nn.LeakyReLU())
        else:
            assert False, "non_lin = {} not valid: 'tanh', 'relu', 'leakyrelu'".format(non_lin)


        self.Block.add_module("Pool", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False))
        if dropout > 0:
            self.Block.add_module("Drop", nn.Dropout2d(dropout))

    def forward(self, x, ):

        if self.skip_connection:
            if not self.first_block:

                x, skip_con_list = x

            else:
                skip_con_list = []

        x = self.Block(x)
        if self.skip_connection:
            if not self.last_block:
                skip_con_list.append(x)
            x = [x, skip_con_list]

        return x

class CNN(nn.Module):
    def __init__(self,
                 social_pooling=False,
                 channels_cnn=4,
                 mlp=32,
                 encoder_h_dim=16,
                 insert_trajectory=False,
                 need_decoder=False,
                 PhysFeature=False,
                 grid_size_in=32,
                 grid_size_out=32,
                 num_layers=3,
                 dropout=0.,
                 batch_norm=False,
                 non_lin_cnn="tanh",
                 in_channels=3,
                 skip_connection=False,
                 ):
        super(CNN, self).__init__()
        self.__dict__.update(locals())


        self.bottleneck_dim = int(grid_size_in / 2 ** (num_layers - 1)) ** 2

        num_layers_dec = int(num_layers + ((grid_size_out - grid_size_in) / grid_size_out))

        self.encoder = nn.Sequential()

        layer_out = channels_cnn
        self.encoder.add_module("ConvBlock_1", Conv_Blocks(in_channels, channels_cnn,
                                                           dropout=dropout,
                                                           batch_norm=batch_norm,
                                                           non_lin=self.non_lin_cnn,
                                                           first_block=True,
                                                           skip_connection=self.skip_connection

                                                           ))
        layer_in = layer_out
        for layer in np.arange(2, num_layers + 1):

            if layer != num_layers:
                layer_out = layer_in * 2
                last_block = False
            else:
                layer_out = layer_in
                last_block = True
            self.encoder.add_module("ConvBlock_%s" % layer,
                                    Conv_Blocks(layer_in, layer_out,
                                                dropout=dropout,
                                                batch_norm=batch_norm,
                                                non_lin=self.non_lin_cnn,
                                                skip_connection=self.skip_connection,
                                                last_block=last_block
                                                ))
            layer_in = layer_out

        self.bootleneck_channel = layer_out
        if self.need_decoder:

            self.decoder = nn.Sequential()
            layer_in = layer_out
            for layer in range(1, num_layers_dec + 1):
                first_block = False
                extra_d = 0
                layer_in = layer_out
                last_block = False
                filter = 4
                padding = 1
                if layer == 1:
                    if self.insert_trajectory:
                        extra_d = 1

                    first_block = True
                    layer_out = layer_in

                else:
                    layer_out = int(layer_in / 2.)

                if layer == num_layers_dec:
                    layer_out = 1
                    last_block = True
                    padding = 0
                    filter = 3

                self.decoder.add_module("UpConv_%s" % layer,
                                        UpConv_Blocks(int(layer_in + extra_d),
                                                      layer_out,
                                                      first_block=first_block,
                                                      filter=filter,
                                                      padding=padding,
                                                      dropout=dropout,
                                                      batch_norm=batch_norm,
                                                      non_lin=self.non_lin_cnn,
                                                      skip_connection=self.skip_connection,
                                                      last_block=last_block))

        if self.insert_trajectory:
            self.traj2cnn = make_mlp(
                dim_list=[encoder_h_dim, mlp, self.bottleneck_dim],
                activation_list=["tanh", "tanh"],
            )

        self.init_weights()

    def init_weights(self):
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                m.bias.data.fill_(0.01)
            # if type(m) in [nn.ConvTranspose2d]:
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
            # m.bias.data.fill_(50)

        def init_xavier(m):
            if type(m) == [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        if self.non_lin_cnn in ['relu', 'leakyrelu']:
            self.apply(init_kaiming)
        elif self.non_lin_cnn == "tanh":
            self.apply(init_xavier)
        else:
            assert False, "non_lin not valid for initialisation"

    def forward(self, image, traj_h=torch.empty(1), pool_h=torch.empty(1)):
        output = {}

        enc = self.encoder(image)

        if self.PhysFeature:
            # enc_out = self.leakyrelu(self.encoder_out(enc))
            # enc_out = enc_out.permute(1, 0, 2, 3).view(1, enc_out.size(0), -1)
            output.update(Features=enc)

        if self.need_decoder:

            if self.skip_connection:
                batch, c, w, h = enc[0].size()
                in_decoder, skip_con_list = enc

            else:
                batch, c, w, h = enc.size()
                in_decoder = enc

            if self.insert_trajectory:

                traj_enc = self.traj2cnn(traj_h)

                traj_enc = traj_enc.view(batch, 1, w, h)
                in_decoder = torch.cat((traj_enc, in_decoder), 1)
            if self.social_pooling:

                social_enc = self.social_states(pool_h)

                social_enc = social_enc.view(batch, 1, w, h)
                in_decoder = torch.cat((social_enc, in_decoder), 1)
            if self.skip_connection: in_decoder = [in_decoder, skip_con_list]
            dec = self.decoder(in_decoder)
            output.update(PosMap=dec)

        return output

class VisualNetwork(nn.Module):
    """VisualNetwork is the parent class for the attention and goal networks generating the CNN"""

    def __init__(self,
                 decoder_h_dim=128,
                 dropout=0.0,
                 batch_norm=False,
                 mlp_dim=32,
                 img_scaling=0.25,
                 final_embedding_dim=4,
                 grid_size_in=16,
                 grid_size_out=16,
                 num_layers=1,
                 batch_norm_cnn=True,
                 non_lin_cnn="relu",
                 img_type="local_image",
                 skip_connection=False,
                 channels_cnn=4,
                 social_pooling=False,
                 **kwargs):

        super(VisualNetwork, self).__init__()
        self.__dict__.update(locals())

    def init_cnn(self):
        self.CNN = CNN(social_pooling=self.social_pooling,
                       channels_cnn=self.channels_cnn,
                       encoder_h_dim=self.decoder_h_dim,
                       mlp=self.mlp_dim,
                       insert_trajectory=True,
                       need_decoder=self.need_decoder,
                       PhysFeature=self.PhysFeature,
                       grid_size_in=self.grid_size_in,
                       grid_size_out=self.grid_size_out,
                       dropout=self.dropout,
                       batch_norm=self.batch_norm_cnn,
                       non_lin_cnn=self.non_lin_cnn,
                       num_layers=self.num_layers,
                       in_channels=4,
                       skip_connection=self.skip_connection
                       )

class GoalGlobal(VisualNetwork):

    def __init__(self,
                 temperature=1, # temperature of the gumbel sampling
                 force_hard=True, # mode of the gumbel sampling
                 **kwargs):

        super(GoalGlobal, self).__init__()
        VisualNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())

        self.PhysFeature = False
        self.need_decoder = True

        self.init_cnn()
        self.gumbelsampler = GumbelSampler(
            temp=self.temperature,
            grid_size_out=self.grid_size_out,
            force_hard=force_hard,
            scaling=self.img_scaling)
    def forward(self, features, h, pool_h=torch.empty(1)):
        cnn_out = self.CNN(features, h, pool_h)

        final_pos, final_pos_map_decoder, final_pos_map, y_softmax, y_scores = self.gumbelsampler(cnn_out)
        return final_pos, final_pos_map_decoder, final_pos_map, y_softmax, y_scores

# Motion Encoder

class MotionEncoder(nn.Module):
    """MotionEncoder extracts dynamic features of the past trajectory and consists of an encoding LSTM network"""

    def __init__(self,
        encoder_h_dim=64,
        input_dim=2,
        embedding_dim=16,
        dropout=0.0):
        """ Initialize MotionEncoder.
        Parameters.
            encoder_h_dim (int) - - dimensionality of hidden state
            input_dim (int) - - input dimensionality of spatial coordinates
            embedding_dim (int) - - dimensionality spatial embedding
            dropout (float) - - dropout in LSTM layer
        """
        super(MotionEncoder, self).__init__()
        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        if embedding_dim:
            self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
            self.encoder = nn.LSTM(embedding_dim, encoder_h_dim)
        else:
            self.encoder = nn.LSTM(input_dim, encoder_h_dim)

    def init_hidden(self, batch, obs_traj):

        return (
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj),
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj)
        )

    def forward(self, obs_traj, state_tuple=None):
        """ Calculates forward pass of MotionEncoder
            Parameters:
                obs_traj (tensor) - - Tensor of shape (obs_len, batch, 2)
                state_tuple (tuple of tensors) - - Tuple with hidden state (1, batch, encoder_h_dim) and cell state tensor (1, batch, encoder_h_dim)
            Returns:
                output (tensor) - - Output of LSTM netwok for all time steps (obs_len, batch, encoder_h_dim)
                final_h (tensor) - - Final hidden state of LSTM network (1, batch, encoder_h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        if not state_tuple:
            state_tuple = self.init_hidden(batch, obs_traj)
        if self.embedding_dim:
            obs_traj = self.spatial_embedding(obs_traj)

        output, state = self.encoder(obs_traj, state_tuple)
        final_h = state[0]
        return output, final_h

# Parameters

dropout: float = 0.0
batch_norm: bool = False
input_dim: int = 2  #
pred_len: int = 12

# Generator dim
encoder_h_dim_g: int = 16
decoder_h_dim_g: int = 16

mlp_dim: int = 32
embedding_dim: int = 8
load_occupancy: bool = False
pretrain_cnn : int = 0

# parameters global goal / attention
temperature_global: int = 1
grid_size_in_global: int = 32
grid_size_out_global: int = 32
num_layers_cnn_global: int = 3
batch_norm_cnn_global: bool = True
dropout_cnn_global: float = 0.3
non_lin_cnn_global: str = "relu"
scaling_global: float = 1.
force_hard_global: bool = True
final_embedding_dim_global: int = 4
skip_connection_global: bool = False
channels_cnn_global: int = 4
global_vis_type: str = "goal"

# parameters routing module
rm_vis_type : str = "attention"
num_layers_cnn_rm: int = 3
batch_norm_cnn_rm=True
dropout_cnn_rm=0.0
non_lin_cnn_rm="relu"
grid_size_rm=16
scaling_local=0.20
force_hard_rm=True
noise_attention_dim_rm=8
final_embedding_dim_rm=4
skip_connection_rm = False
channels_cnn_rm = 4

scaling = (2 * grid_size_in_global + 1) / (2 * grid_size_out_global + 1) * scaling_global

goalmodule = GoalGlobal(
                channels_cnn = channels_cnn_global,
                decoder_h_dim=decoder_h_dim_g,
                dropout=dropout_cnn_global,
                batch_norm=batch_norm,
                mlp_dim=mlp_dim,
                img_scaling=scaling,
                final_embedding_dim=final_embedding_dim_global,
                grid_size_in=grid_size_in_global,
                grid_size_out=grid_size_out_global,
                num_layers=num_layers_cnn_global,
                batch_norm_cnn=batch_norm_cnn_global,
                non_lin_cnn=non_lin_cnn_global,
                temperature = temperature_global,
                force_hard = force_hard_global,
                skip_connection = skip_connection_global)

# Test 

# Check seq_collate!
"""
return {"in_xy": obs_traj,
        "gt_xy": pred_traj,
        "in_dxdy": obs_traj_rel,
        "gt_dxdy": pred_traj_rel,
        "size": torch.LongTensor([obs_traj.size(1)]),
        "scene_img": scene_img_list,
        "global_patch": global_patch,
        "prob_mask": prob_mask,
        "occupancy": wall_list,
        "local_patch": local_patch,
        "seq_start_end": seq_start_end
        }
"""

encoder = MotionEncoder(
            encoder_h_dim=encoder_h_dim_g,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
obs_traj_rel = x = torch.randn(20,10,2) # num_obs x agents x 2 (x|y)
# encoder_out, h_enc = encoder(batch["in_dxdy"])
encoder_out, h_enc = encoder(obs_traj_rel)

# final_pos, final_pos_map_concrete, final_pos_map, y_softmax, y_scores = goalmodule(batch["global_patch"],h_enc)
im_height = im_width = 600
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global_patch = rand(3,im_height,im_width).to(device)
final_pos, final_pos_map_concrete, final_pos_map, y_softmax, y_scores = goalmodule(global_patch,h_enc)