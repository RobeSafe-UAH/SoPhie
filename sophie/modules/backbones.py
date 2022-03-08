from turtle import forward
import torch
from torch import nn

from sophie.utils.load_models import load_VGG_model
from sophie.modules.encoders import EncoderLSTM as Encoder

class VisualExtractor(nn.Module):
    """
    In this class add all the visual extractors that will be used in the architecture.
    Currently supporting:
        - VGG{11, 13, 16, 19}
    """
    def __init__(self, visual_extractor_type, config=None):
        super(VisualExtractor, self).__init__()
        if visual_extractor_type == "vgg19":
            self.module = VGG(config)
        elif visual_extractor_type == "home":
            self.module = HOME()
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

class HOME(nn.Module):

    def __init__(self, ch_list=[3,32,64,128,128]):
        super().__init__()
        assert len(ch_list) == 5, "ch_list must have 5 elements"
        self.m = nn.Sequential(
            nn.Conv2d(ch_list[0],ch_list[1],3,1,1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch_list[1],ch_list[2],3,1,1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch_list[2],ch_list[3],3,1,1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch_list[3],ch_list[4],3,1,1),
            nn.ReLU()
        )

    def forward(self, x):
        """
            x: (b,3,224,224)
            return (b,128,28,28)
        """
        return self.m(x)