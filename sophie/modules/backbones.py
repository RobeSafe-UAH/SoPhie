import torch
from torch import nn

from sophie.utils.load_models import load_VGG_model
from sophie.modules.encoders import Encoder

class JointExtractor(nn.Module):
    """
    In this class add all the joint extractors that will be used in the architecture.
    """

    def __init__(self, joint_extractor_type, config):
        super(JointExtractor, self).__init__()
        
        
        if joint_extractor_type == "encoder_sort":
            self.module = Encoder(**config.encoder)
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
        """
            Input shape: Batch, Channels_in, H_in, W_in
            Output shape: Batch, Channels_out, H_out, W_out
            ?> CHECK DIMENSION BEFORE FORWARD
        """
        image_feature_map = self.module(data_input)
        return image_feature_map


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