import numpy as np
import yaml
import cv2
import time

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

def test_visual_extractor():
    opt = {
        "vgg_type": 19,
        "batch_norm": False,
        "pretrained": True,
        "features": True
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg_19 = VisualExtractor("vgg19", opt).to(device)
    image_test = rand(1,3,600,300).to(device) # batch, channel, H, W
    print(">>> ", vgg_19(image_test).shape) # batch, 512, 18, 9

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


def test_sophie_discriminator():
    """
    """

    batch = 8 # Number of trajectories
    number_of_waypoints = 10 # Waypoints per trajectory 
    points_dim = 2 # xy
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, points_dim)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    with open(r'./configs/sophie.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)

    discriminator = SoPhieDiscriminator(config_file.sophie.discriminator)
    discriminator.build()
    discriminator.to(device)
    discriminator.forward(predicted_trajectory)

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

def test_read_file():
    data = read_file("./data_test.txt", "tab")
    print("data: ", data)
    frames = np.unique(data[:, 0]).tolist()
    print("frames: ", frames)

def test_sophie_generator():

    batch = 8 # Number of trajectories
    number_of_waypoints = 10 # Waypoints per trajectory 
    points_dim = 2 # xy
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, points_dim)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device)

    with open(r'./configs/sophie.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)

    discriminator = SoPhieGenerator(config_file)
    discriminator.build()
    discriminator.to(device)
    discriminator.forward(predicted_trajectory)

def test_dataLoader():
    data = EthUcyDataset("./data/datasets/zara1/train", videos_path="./data/datasets/videos/")
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
    for batch in loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch

        print("> ", obs_traj.shape, obs_traj_rel.shape, seq_start_end.shape)
        #assert 1 == 0, "aiie"

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
    number_of_waypoints = 10 # Waypoints per trajectory 
    batch = 2 # xy
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, dim_features)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    print("Physical attention config: ", config_file.sophie.generator.physical_attention)
    physical_attention_module = SATAttentionModule(config_file.sophie.generator.physical_attention).to(device)
    print("Physical Attention Module: ", physical_attention_module)

    alpha, context_vector = physical_attention_module.forward(visual_extractor_features, predicted_trajectory)

    print("Alpha: ", alpha, alpha.shape)
    print("Context vector: ", context_vector, context_vector.shape)

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
    number_of_waypoints = 10 # Waypoints per trajectory 
    batch = 2 # xy
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, dim_features)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    print("Social attention config: ", config_file.sophie.generator.social_attention)
    social_attention_module = SATAttentionModule(config_file.sophie.generator.social_attention).to(device)
    print("Social Attention Module: ", social_attention_module)

    alpha, context_vector = social_attention_module.forward(joint_extractor_features, predicted_trajectory)

    print("Alpha: ", alpha, alpha.shape)
    print("Context vector: ", context_vector, context_vector.shape)

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

    return pred_traj

def read_video(path, new_shape):
    cap = cv2.VideoCapture(path) 
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_list = []
    while (num_frames > 0):
        _, frame = cap.read()
        print("_ ", _)
        num_frames = num_frames - 1
        re_frame = cv2.resize(frame, new_shape)
        frames_list.append(re_frame)
    cap.release()
    return frames_list

if __name__ == "__main__":
    # test_visual_extractor() # output: batch, 512, 18, 9
    # test_joint_extractor() # output: input_len, batch, hidden_dim
    # test_read_file()
    # test_mlp()
    # test_encoder()
    test_dataLoader()
    # test_decoder()
    # test_physical_attention_model()
    # test_social_attention_model()
    # test_concat_features()
    # test_sophie_discriminator()

    # path_video = "./data/datasets/videos/seq_eth.avi"
    # image_list = read_video(path_video, (600,600))

    # print("image_list: ", type(image_list), len(image_list))
