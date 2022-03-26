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

class Patch_gen():
    def __init__(self, img_scaling=0.5,
                 grid_size=16,
                 type_img="small_image",
                 ):
        self.__dict__.update(locals())

    def get_patch(self, scene_image, last_pos):
        scale = 1. / self.img_scaling
        last_pos_np = last_pos.detach().cpu().numpy()

        image_list = []
        for k in range(len(scene_image)):
            image = scene_image[k][self.type_img]

            center = last_pos_np[k] * scale
            x_center, y_center = center.astype(int)
            cropped_img = image.crop(
                (int(x_center - self.grid_size), int(y_center - self.grid_size), int(x_center + self.grid_size + 1),
                 int(y_center + self.grid_size + 1)))

            cropped_img = -1 + torch.from_numpy(np.array(cropped_img) * 1.) * 2. / 256

            position = torch.zeros((1, self.grid_size * 2 + 1, self.grid_size * 2 + 1, 1))
            position[0, self.grid_size, self.grid_size, 0] = 1.
            image = torch.cat((cropped_img.float().unsqueeze(0), position), dim=3)

            image = image.permute(0, 3, 1, 2)
            image_list.append(image.clone())

        img = torch.cat(image_list)

        img = img.to(last_pos)

        return img

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

class AttentionNetwork(VisualNetwork):
    def __init__(self,
                 noise_attention_dim=8,
                 **kwargs
                 ):
        super(AttentionNetwork, self).__init__()
        VisualNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())
        self.PhysFeature = True
        self.skip_connection = False
        self.need_decoder = False

        self.init_cnn()
        self.final_embedding = self.CNN.bottleneck_dim + self.noise_attention_dim
        attention_dims = [self.CNN.bootleneck_channel, self.mlp_dim, 1]
        activation = ['leakyrelu', None]
        self.cnn_attention = make_mlp(
            attention_dims,
            activation_list=activation, )

    def get_noise(self, batch_size, type="gauss"):
        """
           Create noise vector:
           Parameters
           ----------
           batchsize : int, length of noise vector
           noise_type: str, 'uniform' or 'gaussian' noise
           Returns
           -------
           Random noise vector
           """

        if type == "gauss":
            return torch.randn((1, batch_size, self.noise_attention_dim))
        elif type == "uniform":

            rand_num = torch.rand((1, batch_size, self.noise_attention_dim))
            return rand_num
        else:
            raise ValueError('Unrecognized noise type "%s"' % noise_type)

class AttentionRoutingModule(AttentionNetwork):
    def __init__(self,
                 **kwargs):
        super(AttentionNetwork, self).__init__()
        AttentionNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())
        self.img_patch = Patch_gen(img_scaling=self.img_scaling,
                                   grid_size=self.grid_size_in,
                                   type_img=self.img_type)
        self.init_cnn()

    def forward(self, scene_img, last_pos, h, noise=torch.Tensor()):

        img_patch = self.img_patch.get_patch(scene_img, last_pos)
        visual_features = self.CNN(img_patch, h)["Features"].permute(0, 2, 3, 1)
        batch_size, hh, w, c = visual_features.size()
        visual_features = visual_features.view(batch_size, -1, c)
        attention_scores = self.cnn_attention(visual_features)
        attention_vec = attention_scores.softmax(dim=1).squeeze(2).unsqueeze(0)
        if self.noise_attention_dim > 0:
            if len(noise) == 0:
                noise = self.get_noise(batch_size)
            else:
                assert noise.size(-1) != self.noise_attention_dim, "dimension of noise {} not valid".format(
                    noise.size())

        x = torch.cat((attention_vec, noise.to(attention_vec)), dim=2)

        return x, attention_vec, img_patch, noise

class RoutingModule(nn.Module):
    """RoutingModule is part of TrajectoryGenerator and generates the prediction for each time step.
    The MotionDecoder consists of a LSTM network and a local goal network or attention network"""

    def __init__(
            self,
            seq_len=12,
            input_dim=2,
            decoder_h_dim=128,
            embedding_dim=64,
            dropout=0.0,
            batch_norm=False,
            mlp_dim=32,
            img_scaling_local=0.25,
            final_embedding_dim_rm=4,
            rm_vis_type="attention",
            grid_size_rm=8,
            dropout_cnn_rm=0.0,
            num_layers_rm=3,
            non_lin_cnn_rm="relu",
            force_hard_rm=True,
            temperature_rm=1,
            batch_norm_cnn_rm=False,
            noise_attention_dim_rm=True,
            skip_connection_rm=False,
            channels_cnn_rm=4,
            global_vis_type="goal"):
        """Initialise Motion Decoder network
                Parameters.
                    seq_len (int) - - Prediction length of trajectory
                    input_dim (int) - - input / output dimensionality of spatial coordinates
                    decoder_h_dim (int) - - hidden state dimenstion of decoder LSTM
                    embedding_dim (int) - - dimensionality spatial embedding
                    dropout (float) - - dropout
                    final_embedding_dim (int) - - embedding for final position estimate
                    mlp_dim (int) - - bottleneck dimensionality of mlp networks
                    PhysAtt (bool) - - depreciated. should not be used
                    device (torch.device) - - Choose device: cpu or gpu (cuda)
                    batch_norm (bool) - - if true, applies batch norm in mlp networks
                    img_scaling (float) - - ratio [m/px] between real and pixel space
                    grid_size (int) - - defines size of image path in goal / attention network (grid size is 2xgrid_size +1 )
                    decoder_type ("goal", "attention", none) - -
        """
        super(RoutingModule, self).__init__()

        self.__dict__.update(locals())
        if self.rm_vis_type:
            if self.rm_vis_type == "attention":
                self.rm_attention = AttentionRoutingModule(
                    channels_cnn=self.channels_cnn_rm,
                    decoder_h_dim=self.decoder_h_dim,
                    dropout=self.dropout_cnn_rm,
                    mlp_dim=self.mlp_dim,
                    img_scaling=self.img_scaling_local,
                    grid_size_in=self.grid_size_rm,
                    grid_size_out=self.grid_size_rm,
                    num_layers=self.num_layers_rm,
                    batch_norm_cnn=self.batch_norm_cnn_rm,
                    non_lin_cnn=self.non_lin_cnn_rm,
                    final_embedding_dim=final_embedding_dim_rm,
                    noise_attention_dim=self.noise_attention_dim_rm,
                    skip_connection=self.skip_connection_rm)
                self.final_embedding_dim_rm = self.rm_attention.final_embedding
            self.output_dim = self.decoder_h_dim + self.final_embedding_dim_rm

        elif not self.rm_vis_type:
            self.output_dim = self.decoder_h_dim

        else:
            assert False, "`{}` not valid for `decoder_type`: Choose `goal`, 'attention`, or none".format(decoder_type)

        self.final_output = make_mlp(
            [self.output_dim, self.mlp_dim, self.input_dim],
            activation_list=["relu", None],
            dropout=dropout,
            batch_norm=self.batch_norm)
        self.spatial_embedding = nn.Linear(self.input_dim, self.embedding_dim)

        if self.global_vis_type == "goal":
            self.input_dim_decoder = self.self.embedding_dim * 2 + 1

        else:
            self.input_dim_decoder = self.embedding_dim

        self.decoder = nn.LSTM(self.input_dim_decoder, self.decoder_h_dim)

    def forward(self, last_pos, rel_pos, state_tuple, dist_to_goal=0, scene_img=None):
        """ Calculates forward pass of MotionDecoder
            Parameters:
                obs_traj (tensor) - - Tensor of shape (obs_len, batch, 2)
                state_tuple (tuple of tensors) - - Tuple with hidden state (1, batch, encoder_h_dim) and cell state tensor (1, batch, encoder_h_dim)
            Returns:
                output (tensor) - - Output of LSTM netwok for all time steps (obs_len, batch, encoder_h_dim)
                final_h (tensor) - - Final hidden state of LSTM network (1, batch, encoder_h_dim)
        """

        batch_size = rel_pos.size(0)
        pred_traj_fake_rel = []
        pred_traj_fake = []
        softmax_list = []
        final_pos_list = []
        img_patch_list = []
        final_pos_map_decoder_list = []

        for t in range(self.seq_len):

            decoder_input = self.spatial_embedding(rel_pos)
            decoder_input = decoder_input.view(1, batch_size, self.embedding_dim)
            if self.global_vis_type != "none":
                distance_embeding = self.spatial_embedding(dist_to_goal)
                time_tensor = -1 + 2 * torch.ones(1, decoder_input.size(1), 1) * t / self.seq_len
                time_tensor = time_tensor.to(decoder_input)

                decoder_input = torch.cat((decoder_input, distance_embeding, time_tensor), -1)

            output, state_tuple = self.decoder(decoder_input, state_tuple)

            if self.rm_vis_type == "attention":
                final_emb, y_softmax, img_patch, noise = self.rm_attention(scene_img, last_pos, state_tuple[0])
            else:
                final_emb = torch.Tensor([]).to(state_tuple[0])
                img_patch = []

            input_final = torch.cat((state_tuple[0], final_emb), 2)

            img_patch_list.append(img_patch)

            # rel_pos = final_pos[0]

            rel_pos = self.final_output(input_final)
            rel_pos = rel_pos.squeeze(0)

            curr_pos = rel_pos + last_pos
            dist_to_goal = dist_to_goal - rel_pos
            pred_traj_fake_rel.append(rel_pos.clone().view(batch_size, -1))
            pred_traj_fake.append(curr_pos.clone().view(batch_size, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake = torch.stack(pred_traj_fake, dim=0)

        output = {"out_xy": pred_traj_fake,
                  "out_dxdy": pred_traj_fake_rel,
                  "h": state_tuple[0]}

        if self.rm_vis_type == "attention":
            output.update({"image_patches": torch.stack(img_patch_list, dim=0)})

        return output

print("Test Decoder")
print(RoutingModule())

# Generator forward

# OBJECTIVE: LINK OUR DECODER (SOPHIE) WITH OUT GOAL PROPOSALS!!!!!!!!!!!!!!!!!!!!!!1

"""
def forward(self, batch, final_pos_in=torch.Tensor()):
    batch_size = batch["in_xy"].size(1)
    encoder_out, h_enc = self.encoder(batch["in_dxdy"])

    final_pos, final_pos_map_concrete, final_pos_map, y_softmax, y_scores = self.goalmodule(batch["global_patch"],h_enc)
    out = {
        "y_map": final_pos_map_concrete,
        "y_softmax": y_softmax,
        "final_pos": final_pos,
        "y_scores": y_scores
    }
    if self.mode == "pretrain":
        return out

    if len(final_pos_in) > 0:
        final_pos = final_pos_in

    final_pos_embedded = self.final_pos_embedding(final_pos)
    h = self.encoder2decoder(torch.cat((h_enc, final_pos_embedded), 2))
    c = self.init_c(batch_size).to(batch["in_xy"])

    # last position
    x0 = batch["in_xy"][-1]
    v0 = batch["in_dxdy"][-1]
    state_tuple = (h, c)
    out.update(self.routingmodule(last_pos=x0, dist_to_goal=final_pos, rel_pos=v0, state_tuple=state_tuple,
                            scene_img=batch["scene_img"]))

    return {**out,
            "h_encoder": h_enc,
            }
"""