import torch
from torch import nn

from sophie.modules.decoders import Decoder
from sophie.modules.encoders import Encoder
from sophie.modules.attention import PhysicalAttention, SocialAttention
from sophie.modules.layers import VisualExtractor, JointExtractor, MLP

"""
SoPhie implementation based on:
    - https://arxiv.org/pdf/1806.01482.pdf
"""

class SoPhieGenerator(nn.Module):
    def __init__(self, config):
        super(SoPhieGenerator, self).__init__()
        self.config = config

    def build(self):
        self._build_feature_extractor_modules()
        self._build_Attention_modules()
        self._build_decoder_module()

    def _build_feature_extractor_modules(self):
        self.visual_feature_extractor = VisualExtractor(self.config.generator.visual_features)
        self.joint_feature_extractor = JointExtractor(self.config.generator.joint_features)

    def _build_Attention_modules(self):
        self.physical_attention = PhysicalAttention(self.config.generator.physical_attention)
        self.social_attention = SocialAttention(self.config.generator.social_attention)

    def _build_decoder_module(self):
        self.generator_decoder = Decoder(self.config.generator.decoder)

    def process_visual_feature(self, *args):
        """
        Define
        """
        return 1

    def process_joint_feature(self, *args):
        """
        Define
        """
        return 1

    def process_physical_attention(self, *args):
        """
        Define
        """
        return 1

    def process_social_attention(self, *args):
        """
        Define
        """
        return 1

    def process_attention(self, *args):
        """
        Define
        """
        return 1

    def add_white_noise(self, *args):
        """
        Define
        """
        return 1

    def process_decoder_gan(self, *args):
        """
        Define
        """
        return 1

    def forward(self, sample):
        """
        Define
        """
        visual_feat = self.process_visual_feature(sample)
        joint_feat = self.process_joint_feature(sample)
        attention_features = self.process_attention(visual_feat, joint_feat)
        attention_features_noise = self.add_white_noise(attention_features)
        pred_traj = self.process_decoder_gan(attention_features_noise)
        return pred_traj


class SoPhieDiscriminator(nn.Module):

    def __init__(self, config):
        super(SoPhieDiscriminator, self).__init__()
        self.config = config

    def build(self):
        self.encoder_discriminator = self._build_encoder_module()
        self.classifier_discriminator = self._build_classifier_module()

    def _build_encoder_module(self):
        return Encoder(**self.config.discriminator.encoder)

    def _build_classifier_module(self):
        return MLP(**self.config.classifier)

    def process_encoder_gan(self, *args):
        """
        Define
        """
        return 1

    def forward(self, traj):
        traj_classified = self.process_encoder_gan(traj)
        return traj_classified

