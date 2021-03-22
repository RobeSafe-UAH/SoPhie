import torch
from torch import nn

from sophie.utils.load_models import load_VGG_model
from sophie.modules.encoders import Encoder

class MLP(nn.Module):

    def __init__(self, dim_list, activation='relu', batch_norm=True, dropout=0):
        super(MLP, self).__init__()
        self.module = self.make_mlp(dim_list, activation, batch_norm, dropout)

    def make_mlp(self, dim_list, activation='relu', batch_norm=True, dropout=0):
        layers = []
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, data_input):
        return self.module(data_input)

class JointExtractor(nn.Module):
    """
    In this class add all the joint extractors that will be used in the architecture.
    """

    def __init__(self, joint_extractor_type, **config):
        super(JointExtractor, self).__init__()
        
        # TODO
        if joint_extractor_type == "TYPE_A":
            self.module = lambda a, b: a+b
        else:
            raise NotImplementedError("Unknown joint extractor module {}.".format(joint_extractor_type))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class VisualExtractor(nn.Module):
    """
    In this class add all the visual extractors that will be used in the architecture.
    Currently supporting:
        - VGG{11, 13, 16, 19}
    """
    def __init__(self, visual_extractor_type, config):
        super(VisualExtractor, self).__init__()
        if visual_extractor_type == "vgg19":
            self.module = VGG(config)
        else:
            raise NotImplementedError("Unknown visual extractor module {}.".format(visual_extractor_type))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VGG(nn.Module):

    def __init__(self, config):
        super(VGG, self).__init__()
        self.module = load_VGG_model(**config)

    def forward(self, data_input):
        return 1


class SortJointExtractor(nn.Module):

    def __init__(self, config):
        super(SortJointExtractor, self).__init__()
        self.encoder = Encoder(**config.encoder)

    def sort_method(self, data):
        return 1
        
    def forward(self, data_input):
        encoder_output = self.encoder(data_input)
        joint_features = self.sort_method(encoder_output)
        return joint_features