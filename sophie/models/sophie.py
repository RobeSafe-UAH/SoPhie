import torch
from torch import nn

from sophie.modules.decoders import Decoder
from sophie.modules.encoders import Encoder
from sophie.modules.classifiers import Classifier
from sophie.modules.attention import SATAttentionModule
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

    def set_num_agents(self, num_agents):
        self.num_agents = num_agents

    def build(self):
        self._build_feature_extractor_modules()
        self._build_Attention_modules()
        self._build_decoder_module()

    def _build_feature_extractor_modules(self):
        self.visual_feature_extractor = VisualExtractor(
            self.config.visual_extractor.type,
            self.config.visual_extractor.vgg
        )

        for param in self.visual_feature_extractor.parameters():
            param.requires_grad = False

        self.joint_feature_extractor = JointExtractor(
            self.config.joint_extractor.type,
            self.config.joint_extractor.config
        )

    def _build_Attention_modules(self):
        self.physical_attention = SATAttentionModule(self.config.physical_attention)
        self.social_attention = SATAttentionModule(self.config.social_attention)

    def _build_decoder_module(self):
        self.generator_decoder = Decoder(self.config.decoder)

    def process_visual_feature(self, image):
        visual_features = self.visual_feature_extractor(image)
        return visual_features

    def process_joint_feature(self, trajectories):
        joint_features, a = self.joint_feature_extractor(trajectories)
        return joint_features

    def process_physical_attention(self, visual_feature, decoder_state):
        """
        We only care the second element of the tuple (physical features = physical context vector)
        """
        _, physical_features = self.physical_attention(visual_feature, decoder_state, self.num_agents)
        return physical_features

    def process_social_attention(self, joint_feature, decoder_state):
        """
        We only care the second element of the tuple (social features = social context vector)
        """
        _, social_features = self.social_attention(joint_feature, decoder_state, self.num_agents)
        return social_features

    def create_white_noise(self, noise_type, dims):
        if noise_type == "gauss":
            return torch.randn((dims[0], dims[1])).cuda()
            # return torch.randn((dims[0], dims[1]))
        elif noise_type == "uniform":
            rand_num = torch.rand((dims[0], dims[1])).cuda()
            # rand_num = torch.rand((dims[0], dims[1]))
            return rand_num
        else:
            raise ValueError('Unrecognized noise type "%s"' % noise_type)

    def add_white_noise(self, features, white_noise):
        feature_noise = features.add(white_noise)
        return feature_noise 

    def process_decoder(self, features):
        trajectories, final_state = self.generator_decoder(features, self.num_agents)
        return trajectories, final_state

    def forward(self, image, trajectories, num_agents=32):
        """
        Define
        """

        batch = trajectories.shape[1]
        batch_size = int(trajectories.shape[1]/num_agents)

        decoder_state = self.generator_decoder.init_hidden(batch) # hidden_state
        visual_feat = self.process_visual_feature(image)
        joint_feat = self.process_joint_feature(trajectories)

        attention_visual_features = self.process_physical_attention(visual_feat, decoder_state[0]) # decoder_state[0] = hidden_state
        # print("Attention physical features: ", attention_visual_features.shape)
        attention_social_features = self.process_social_attention(joint_feat, decoder_state[0])
        # print("Attention social features: ", attention_social_features.shape)
        attention_features = torch.matmul(attention_visual_features, attention_social_features.t())
        # attention_features = torch.cat((attention_visual_features, attention_social_features), 0)
 
        shape_features = attention_features.shape
        noise = self.create_white_noise(
            self.config.noise.noise_type,
            shape_features
        )
        features_noise = self.add_white_noise(attention_features, noise)
        # print("Features noise: ", features_noise.shape)
        pred_traj, _ = self.process_decoder(features_noise)
        # print("Pred trajectories: ", pred_traj, pred_traj.shape)

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
        _, final_hidden = self.encoder(predicted_trajectory) 
        return final_hidden

    def process_classifier(self, encoded_trajectory):
        classified_trajectory = self.classifier(encoded_trajectory)
        return classified_trajectory

    def forward(self, predicted_trajectory): # Either groundtruth or generated
        """
        Define
        """

        final_hidden = self.process_encoder(predicted_trajectory) # 1, batch, 64
        classified_trajectory = self.process_classifier(final_hidden.squeeze())
        
        return classified_trajectory