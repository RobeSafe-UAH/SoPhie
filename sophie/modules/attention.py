import torch
from torch import nn

class SoftAttentionModule(nn.Module):

    def __init__(self, config):
        super(SoftAttentionModule, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.phase_one = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size)
        )

        self.phase_two = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

        self.phase_three = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features)
        )

    def forward(self, feature_1, feature_2):
        output_1 = self.phase_one(feature_1)
        output_2 = self.phase_two
        output_3 = self.phase_three


class SATAttentionModule(nn.Module):
    
    def __init__(self, config):
        super(SATAttentionModule, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.linear = nn.Linear(config.linear.in_features, config.linear.out_features)
        self.relu  = nn.ReLU()
        self.softmax = nn.Softmax(config.softmax.dim)

    def forward(self, feature_1, feature_decoder):
        linear_output = self.linear(feature_decoder)
        relu_output = self.relu(feature_1 + linear_output)
        alpha = self.softmax(relu_output)
        context = torch.matmul(feature_1, alpha)
        return alpha, context


# Soft attention
class PhysicalAttention(nn.Module):

    def __init__(self, config):
        super(PhysicalAttention, self).__init__()


# Soft attention
class SocialAttention(nn.Module):

    def __init__(self, config):
        super(SocialAttention, self).__init__()