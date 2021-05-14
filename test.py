import numpy as np
import yaml

from types import SimpleNamespace
from sophie.models import SoPhieDiscriminator, SoPhieGenerator
from sophie.modules.layers import MLP
from sophie.modules.backbones import VisualExtractor, JointExtractor
from sophie.modules.encoders import Encoder
from sophie.modules.classifiers import Classifier
from sophie.data_loader.ethucy.dataset import read_file, EthUcyDataset, seq_collate
from sophie.modules.decoders import Decoder
from sophie.modules.attention import SATAttentionModule

from prodict import Prodict

import torch
from torch import nn
from torch import rand
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(r'./configs/sophie.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)

# Read data

def test_read_file():
    data = read_file("./data_test.txt", "tab")
    print("data: ", data)
    frames = np.unique(data[:, 0]).tolist()
    print("frames: ", frames)

def test_dataLoader():
    data = EthUcyDataset("./data/datasets/eth/train")
    print(data)
    batch_size = 64
    loader_num_workers = 4
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate)

    print("loader: ", loader)

# Extractors

def test_visual_extractor():
    opt = {
        "vgg_type": 19,
        "batch_norm": False,
        "pretrained": True,
        "features": True
    }


    vgg_19 = VisualExtractor("vgg19", opt).to(device)
    image_test = rand(1,3,600,600).to(device) # batch, channel, H, W
    print(">>> ", vgg_19(image_test).shape) # batch, 512, 18, 18

def test_joint_extractor():
    opt = {
        "encoder": {
            "num_layers": 1,
            "hidden_dim": 32,
            "emb_dim": 2,
            "dropout": 0,
            "mlp_config": {
                "dim_list": [2, 32],
                "activation": 'relu',
                "batch_norm": False,
                "dropout": 0
            }
        }
    }
    opt = Prodict.from_dict(opt)

    joint_extractor = JointExtractor("encoder_sort", opt).to(device)
    input_trajectory = 10 * np.random.randn(10, 8, 2)
    input_trajectory = torch.from_numpy(input_trajectory).to(device).float()
    print("input_trajectory: ", input_trajectory.shape)

    joint_features, _ = joint_extractor(input_trajectory)

    print("joint_features: ", joint_features.shape)

# Attention modules 

def test_physical_attention_model():
    """
    """

    # Feature 1 (Visual Extractor)

    batch = 1
    channels = 512
    width = 18
    height = 18
    visual_extractor_features = 10 * np.random.randn(batch, channels, width, height)
    visual_extractor_features = torch.from_numpy(visual_extractor_features).to(device).float()

    # Feature decoder

    dim_features = 128 # dim_features
    number_of_waypoints = 12 # Timesteps we are attempting to predict
    batch = 32 # Number of agents
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, dim_features)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    print("Physical attention config: ", config_file.sophie.generator.physical_attention)
    physical_attention_module = SATAttentionModule(config_file.sophie.generator.physical_attention).to(device)
    print("Physical Attention Module: ", physical_attention_module)

    alpha, context_vector = physical_attention_module.forward(visual_extractor_features, predicted_trajectory)

    # print("Alpha: ", alpha, alpha.shape)
    # print("Context vector: ", context_vector, context_vector.shape)
    print("Alpha: ", alpha.shape)
    print("Context vector: ", context_vector.shape)

    return context_vector

def test_social_attention_model():
    """
    """
    print("\n")
    # Feature 1 (Joint Extractor)

    length = 8 # How many previous timesteps we observe for each agent
    batch = 32 # Number of agents
    hidden_dim = 32 # Features dimension
 
    joint_extractor_features = 10 * np.random.randn(length, batch, hidden_dim)
    joint_extractor_features = torch.from_numpy(joint_extractor_features).to(device).float()

    # Feature decoder

    dim_features = 128 # dim_features
    number_of_waypoints = 12 # Timesteps we are attempting to predict
    batch = 32 # Number of agents
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, dim_features)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    print("Social attention config: ", config_file.sophie.generator.social_attention)
    social_attention_module = SATAttentionModule(config_file.sophie.generator.social_attention).to(device)
    print("Social Attention Module: ", social_attention_module)

    alpha, context_vector = social_attention_module.forward(joint_extractor_features, predicted_trajectory)

    # print("Alpha: ", alpha, alpha.shape)
    # print("Context vector: ", context_vector, context_vector.shape)
    print("Alpha: ", alpha.shape)
    print("Context vector: ", context_vector.shape)

    return context_vector

def test_concat_features():
    """
    """

    physical_context_vector = test_physical_attention_model()
    social_context_vector = test_social_attention_model()

    attention_features = torch.cat((physical_context_vector, social_context_vector), 0).to(device)
    generator = SoPhieGenerator(config_file.sophie.generator)

    generator.build()
    generator.to(device)

    shape_features = attention_features.shape
    noise = generator.create_white_noise(
        generator.config.noise.noise_type,
        shape_features
    )

    features_noise = generator.add_white_noise(attention_features, noise)
    pred_traj = generator.process_decoder(features_noise)

    # print("Pred trajectories: ", pred_traj, len(pred_traj))

    return pred_traj

# Multi-Layer Perceptron

def test_mlp():
    opt = {
        "dim_list": [2, 64],
        "activation": 'relu',
        "batch_norm": False,
        "dropout": 0
    }
    opt = Prodict.from_dict(opt)
    mlp = MLP(**opt)
    print(mlp)

# Encoder

def test_encoder():
    opt = {
        "num_layers": 1,
        "hidden_dim": 32,
        "emb_dim": 2,
        "dropout": 0.4,
        "mlp_config": {
            "dim_list": [2, 16],
            "activation": 'relu',
            "batch_norm": False,
            "dropout": 0.4
        }
    }
    encoder = Encoder(**opt)
    print(encoder)

# Decoder

def test_decoder():

    with open(r'./configs/sophie.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)

    input_data = np.random.randn(688,2)
    input_data = torch.from_numpy(input_data).to(device).float()

    print(">>>: ", config_file.sophie.generator.decoder)
    sophie_decoder = Decoder(config_file.sophie.generator.decoder).to(device)
    print("sophie_decoder: ", sophie_decoder)
    trajectories, state_tuple_1 = sophie_decoder(input_data)
    print("trajectories: ", trajectories.shape)
    #print("state_tuple_1: ", state_tuple_1)

# GAN generator

def test_sophie_generator():

    # Image

    width = 600
    height = 600
    channels = 3
    batch_img = 1
    image_test = torch.rand(batch_img, channels, height, width).to(device) # batch, channel, H, W

    # Trajectories

    batch = 32 # Number of trajectories
    number_of_waypoints = 8 # Waypoints per trajectory 
    points_dim = 2 # xy
    trajectories_test = 10 * np.random.randn(number_of_waypoints, batch, points_dim)
    trajectories_test = torch.from_numpy(trajectories_test).to(device).float()

    # Decoder output

    dim_features = 128 # dim_features
    number_of_waypoints = 12 # Timesteps we are attempting to predict
    batch = 32 # Number of agents
    decoder_output_test = 10 * np.random.randn(number_of_waypoints, batch, dim_features)
    decoder_output_test = torch.from_numpy(decoder_output_test).to(device).float()

    sample_dict = {'image' : image_test, 'trajectories' : trajectories_test, 'decoder_output' : decoder_output_test}
    sample = Prodict.from_dict(sample_dict)

    batch_traj = sample.trajectories.shape
    print("Batch: ", batch_traj)

    generator = SoPhieGenerator(config_file.sophie.generator)
    generator.build()
    generator.to(device)
    generator.forward(sample)

# GAN discriminator

def test_sophie_discriminator():
    """
    """

    batch = 32 # Number of trajectories
    number_of_waypoints = 8 # Waypoints per trajectory 
    points_dim = 2 # xy
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, points_dim)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    discriminator = SoPhieDiscriminator(config_file.sophie.discriminator)
    discriminator.build()
    discriminator.to(device)
    discriminator.forward(predicted_trajectory)

if __name__ == "__main__":
    # test_read_file()
    # test_dataLoader()
    # test_visual_extractor() 
    test_joint_extractor() 
    # test_physical_attention_model()
    # test_social_attention_model()
    # test_concat_features()
    # test_mlp()
    # test_encoder()
    # test_decoder()
    # test_sophie_generator()
    # test_sophie_discriminator()
    