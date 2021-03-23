import numpy as np

from types import SimpleNamespace
from sophie.modules.layers import MLP
from sophie.modules.backbones import VisualExtractor
from sophie.models.sophie import SoPhieDiscriminator
from sophie.modules.encoders import Encoder
from sophie.data_loader.ethucy.dataset import read_file


def test_visual_extractor():
    opt = {
        "vgg_type": 19,
        "batch_norm": False,
        "pretrained": True,
        "features": True
    }

    vgg_19 = VisualExtractor("vgg19", opt)
    print(">>> ", vgg_19)


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
    return 1


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

if __name__ == "__main__":
    test_visual_extractor()
