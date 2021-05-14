import torch
from torch import nn

from sophie.modules.decoders import Decoder
from sophie.modules.encoders import Encoder
from sophie.modules.classifiers import Classifier
from sophie.modules.attention import PhysicalAttention, SocialAttention, SATAttentionModule
from sophie.modules.layers import MLP
from sophie.modules.backbones import VisualExtractor, JointExtractor

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
        self.visual_feature_extractor = VisualExtractor(
            self.config.visual_extractor.type,
            self.config.visual_extractor.vgg
        )
        # print("visual_feature_extractor", self.visual_feature_extractor)
        self.joint_feature_extractor = JointExtractor(
            self.config.joint_extractor.type,
            self.config.joint_extractor.config
        )
        # print("joint_feature_extractor", self.joint_feature_extractor)

    def _build_Attention_modules(self):
        self.physical_attention = SATAttentionModule(self.config.physical_attention)
        # print("physical_attention", self.physical_attention)
        self.social_attention = SATAttentionModule(self.config.social_attention)
        # print("social_attention", self.social_attention)

    def _build_decoder_module(self):
        self.generator_decoder = Decoder(self.config.decoder)
        # print("generator_decoder", self.generator_decoder)

    def process_visual_feature(self, image):
        visual_features = self.visual_feature_extractor(image)
        return visual_features

    def process_joint_feature(self, trajectories):
        joint_features = self.joint_feature_extractor(trajectories)
        return joint_features

    def process_physical_attention(self, visual_feature, decoder_state):
        physical_features = self.physical_attention(visual_feature, decoder_state)
        return physical_features

    def process_social_attention(self, joint_feature, decoder_state):
        social_features = self.social_features(joint_feature, decoder_state)
        return social_features

    def create_white_noise(self, noise_type, dims):
        if noise_type == "gauss":
            return torch.randn((dims[0], dims[1])).cuda()
        elif noise_type == "uniform":
            rand_num = torch.rand((dims[0], dims[1])).cuda()
            return rand_num
        else:
            raise ValueError('Unrecognized noise type "%s"' % noise_type)

    def add_white_noise(self, features, white_noise):
        feature_noise = features.add(white_noise)
        return feature_noise 

    def process_decoder(self, features):
        trajectories, final_state = self.generator_decoder(features)
        return trajectories, final_state

    def forward(self, sample):
        """
        Define
        """
        batch = sample.trajectories[1]
        decoder_state = self.generator_decoder.init_hidden(batch)
        visual_feat = self.process_visual_feature(sample.image)
        joint_feat = self.process_joint_feature(sample.trajectories)
        attention_visual_features = self.process_physical_attention(visual_feat, decoder_state)
        attention_social_features = self.process_social_attention(joint_feat, decoder_state)
        attention_features = torch.cat((attention_visual_features, attention_social_features), 0)
        noise = self.create_white_noise(
            self.config.noise.noise_type,
            self.config.noise.dims
        )
        features_noise = self.add_white_noise(attention_features, noise)
        pred_traj = self.process_decoder(features_noise)
        return pred_traj

class SoPhieDiscriminator(nn.Module):

    def __init__(self, config):
        super(SoPhieDiscriminator, self).__init__()
        self.config = config

    def build(self):
        self.encoder = self._build_encoder_module()
        self.classifier = self._build_classifier_module()

    def _build_encoder_module(self):
        return Encoder(**self.config.encoder)

    def _build_classifier_module(self):
        return Classifier(**self.config.classifier)

    def process_encoder(self, predicted_trajectory):
        encoded_trajectory, _ = self.encoder(predicted_trajectory) 
        return encoded_trajectory

    def process_classifier(self, encoded_trajectory):
        classified_trajectory = self.classifier(encoded_trajectory)
        return classified_trajectory

    def forward(self, predicted_trajectory): # Either groundtruth or generated
        """
        Define
        """

        encoded_trajectory = self.process_encoder(predicted_trajectory)
        classified_trajectory = self.process_classifier(encoded_trajectory)

        print("Classified trajectories: ", classified_trajectory)
        
        return classified_trajectory