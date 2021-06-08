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
from sophie.data_loader.ethucy.dataset import read_file, EthUcyDataset, seq_collate, seq_collate_image
from sophie.data_loader.aiodrive.dataset import AioDriveDataset, seq_collate_image_aiodrive
from sophie.modules.decoders import Decoder
from sophie.modules.attention import SATAttentionModule

from prodict import Prodict

import torch
from torch import nn
from torch import rand
from torch.utils.data import DataLoader

# Global variables

## CUDA device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Configuration file

with open(r'./configs/sophie.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)

## Image

im_batch = 1
im_width = 600
im_height = 600

## Trajectories

po = 8 # Past Observations
fo = 12 # Future Observations
na = 32 # Number of Agents 
dlb = 8 # DataLoader batch
tfd = 2 # Trajectory Features Dimension
    
# Read data

def test_read_file():
    data = read_file("./data_test.txt", "tab")
    print("data: ", data)
    frames = np.unique(data[:, 0]).tolist()
    print("frames: ", frames)

def test_dataLoader():
    data = EthUcyDataset("./data/datasets/zara1/train")
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
    print("device: ", device)
    t0 = time.time()
    for batch in loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch

        print("> ", obs_traj.shape, obs_traj_rel.shape, seq_start_end.shape)
        t1 = time.time()
        while(t1 - t0 < 120):
            print(t1-t0)
            t1 = time.time()
        #assert 1 == 0, "aiie"

def test_dataLoader_img():
    data = EthUcyDataset("./data/datasets/zara1/train", videos_path="./data/datasets/videos/")
    print(data)
    batch_size = 64
    loader_num_workers = 4
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate_image)

    print("loader: ", loader)
    print("device: ", device)
    t0 = time.time()
    for batch in loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end, frames) = batch

        print("> ", obs_traj.shape, pred_traj_gt.shape, frames.shape)
        # t1 = time.time()
        # while(t1 - t0 < 120):
        #     print(t1-t0)
        #     t1 = time.time()
        #assert 1 == 0, "aiie"

# Extractors

def test_visual_extractor():
    # opt = {
    #     "vgg_type": 19,
    #     "batch_norm": False,
    #     "pretrained": True,
    #     "features": True
    # }

    # vgg_19 = VisualExtractor("vgg19", opt).to(device)

    vgg_19 = VisualExtractor("vgg19", config_file.sophie.generator.visual_extractor.vgg).to(device)
    image_test = rand(im_batch,3,im_height,im_width).to(device) # batch, channel, H, W
    print(">>> ", vgg_19(image_test).shape) # batch, 512, 18, 18

def test_joint_extractor():
    initial_trajectory = 10 * np.random.randn(po, na, tfd)
    initial_trajectory = torch.from_numpy(initial_trajectory).to(device).float()

    opt = {
        "encoder": {
            "num_layers": 1,
            "hidden_dim": 32,
            "emb_dim": 2,
            "mlp_config": {
                "dim_list": [2, 32],
                "activation": 'relu',
                "batch_norm": False,
                "dropout": 0
            },
            "dropout": 0,
        }
    }

    opt = Prodict.from_dict(opt)

    joint_extractor = JointExtractor("encoder_sort", config_file.sophie.generator.joint_extractor.config).to(device)
    # joint_extractor = JointExtractor("encoder_sort", opt).to(device)
    joint_features, _ = joint_extractor(initial_trajectory)

    print("Joint Features: ", joint_features.shape)

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

    # Hidden Decoder Features

    decoder_dim_features = 128 # dim_features
    predicted_trajectory = 10 * np.random.randn(1, na*dlb, decoder_dim_features)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    physical_attention_module = SATAttentionModule(config_file.sophie.generator.physical_attention).to(device)

    alpha, context_vector = physical_attention_module.forward(visual_extractor_features, predicted_trajectory)

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

    # Hidden Decoder Features

    decoder_dim_features = 128 # dim_features
    predicted_trajectory = 10 * np.random.randn(1, na*dlb, decoder_dim_features)
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
    print("Attention features: ", attention_features.shape)
    generator = SoPhieGenerator(config_file.sophie.generator)

    generator.build()
    generator.to(device)

    shape_features = attention_features.shape
    noise = generator.create_white_noise(
        generator.config.noise.noise_type,
        shape_features
    )

    features_noise = generator.add_white_noise(attention_features, noise)
    pred_traj, _ = generator.process_decoder(features_noise)

    print("Pred trajectories: ", pred_traj, pred_traj.shape, type(pred_traj))

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

    input_data = np.random.randn(768,2)
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
    batch_img = 1*dlb
    image_test = torch.rand(batch_img, channels, height, width).to(device) # batch, channel, H, W

    # Trajectories

    trajectories_test = 10 * np.random.randn(po, na*dlb, tfd)
    trajectories_test = torch.from_numpy(trajectories_test).to(device).float()

    # sample_dict = {'image' : image_test, 'trajectories' : trajectories_test, 'decoder_output' : decoder_output_test}
    # sample_dict = {'image' : image_test, 'trajectories' : trajectories_test}
    # sample = Prodict.from_dict(sample_dict)
    config_file.sophie.generator.decoder.linear_3.input_dim = config_file.dataset.batch_size*2*config_file.sophie.generator.social_attention.linear_decoder.out_features
    config_file.sophie.generator.decoder.linear_3.output_dim = config_file.dataset.batch_size*na

    generator = SoPhieGenerator(config_file.sophie.generator)
    generator.build()
    generator.to(device)
    generator.forward(image_test,trajectories_test)

# GAN discriminator

def test_sophie_discriminator():
    """
    """

    trajectories = 10 * np.random.randn(po+fo, na*dlb, tfd) # 8, 32, 2
    trajectories = torch.from_numpy(trajectories).to(device).float()

    discriminator = SoPhieDiscriminator(config_file.sophie.discriminator)
    discriminator.build()
    discriminator.to(device)
    discriminator.forward(trajectories)

def test_aiodrive_dataset():
    data = AioDriveDataset("./data/datasets/aiodrive_Car/train")
    print(data)
    assert 1 == 0, "aieeee"
    batch_size = 64
    loader_num_workers = 4
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate_image)

    print("loader: ", loader)
    print("device: ", device)
    t0 = time.time()
    for batch in loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end, frames) = batch

        print("> ", obs_traj.shape, pred_traj_gt.shape, frames.shape)
        # t1 = time.time()
        # while(t1 - t0 < 120):
        #     print(t1-t0)
        #     t1 = time.time()
        #assert 1 == 0, "aiie"


if __name__ == "__main__":
    # test_read_file()
    # test_dataLoader()
    # test_visual_extractor() 
    # test_joint_extractor() 
    # test_physical_attention_model()
    # test_social_attention_model()
    # test_concat_features()
    # test_mlp()
    # test_encoder()
    # test_decoder()
    #test_sophie_generator()
    #test_sophie_discriminator()
    test_aiodrive_dataset()

    # path_video = "./data/datasets/videos/seq_eth.avi"
    # image_list = read_video(path_video, (600,600))

    # print("image_list: ", type(image_list), len(image_list))
