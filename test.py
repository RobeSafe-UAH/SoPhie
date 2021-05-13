import numpy as np
import yaml

from types import SimpleNamespace
from sophie.models import SoPhieDiscriminator, SoPhieGenerator
from sophie.modules.layers import MLP
from sophie.modules.backbones import VisualExtractor
from sophie.modules.encoders import Encoder
from sophie.modules.classifiers import Classifier
from sophie.data_loader.ethucy.dataset import read_file, EthUcyDataset, seq_collate

from prodict import Prodict

import torch
from torch import nn
from torch import rand
from torch.utils.data import DataLoader

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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


def test_mlp():
    opt = {
        "dim_list": [2, 64],
        "activation": 'relu',
        "batch_norm": False,
        "dropout": 0
    }
    mlp = MLP(**opt)
    print(mlp)


def test_sophie_discriminator():
    """
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch = 8 # Number of trajectories
    number_of_waypoints = 10 # Waypoints per trajectory 
    points_dim = 2 # xy
    predicted_trajectory = 10 * np.random.randn(number_of_waypoints, batch, points_dim)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device)

    with open(r'./configs/sophie.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)

    discriminator = SoPhieDiscriminator(config_file)
    discriminator.build()
    discriminator.to(device)
    discriminator.forward(predicted_trajectory)

    """
    opt_encoder = { # LSTM based encoder
        "num_layers": 1, # 1 LSTM
        "hidden_dim": 64, # 64 hidden states
        "emb_dim": 16, # Embedded dimension (the output dim of previous MLP is the embedding input of LSTM layer)
        "dropout": 0.4,
        "mlp_config": {
            "dim_list": [2, 16], # Input dim?
            "activation": 'relu',
            "batch_norm": False,
            "dropout": 0.4
        }
    }
    encoder_discriminator = Encoder(**opt_encoder)
    # Classifier
    
    # ?¿?¿ Another MLP for the discriminator ?
    opt_classifier = { # LSTM based encoder
        "softmax_dim": 1,
        "mlp_config": {
            "dim_list": [2, 16], # Input dim?
            "activation": 'relu',
            "batch_norm": False,
            "dropout": 0.4
        }
    }
    softmax_dim = 1
    classifier_discriminator = Classifier(**opt_classifier)
    
    print("Discriminator: ")
    print("Encoder: ", encoder_discriminator)
    print("Classifier: ", classifier_discriminator)
    """

def test_encoder():
    opt = {
        "num_layers": 1,
        "hidden_dim": 32,
        "emb_dim": 16,
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    data = EthUcyDataset("/home/fkite/git-personal/SoPhie/data/datasets/eth/train")
    print(data)
    batch_size = 64
    loader_num_workers = 4
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate)

    print("laoder: ", loader)

if __name__ == "__main__":
    test_visual_extractor()
    #test_sophie_discriminator()
    # test_read_file()
    # test_mlp()
    # test_encoder()
    test_dataLoader()