import torch
from torch import nn

class SATAttentionModule(nn.Module):
    
    def __init__(self, config):
        super(SATAttentionModule, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.linear_decoder = nn.Linear(
            self.config.linear_decoder.in_features, self.config.linear_decoder.out_features
        )
        self.linear_feature = nn.Linear(
            self.config.linear_feature.in_features, self.config.linear_feature.out_features
        )
        self.relu  = nn.ReLU()
        self.softmax = nn.Softmax(self.config.softmax.dim)

    def forward(self, feature_1, feature_decoder):
        """
        Inputs:
            - feature_1: Visual Extractor output (4D {batch, 512, 18, 18}) or Joint Extractor output (3D {L, batch, dim_features})
            - feature_decoder: Generator output (LSTM based decoder; 3D {batch, L, dim_features})
        Outputs:
            - alpha?
            - context_vector: (For physical attention, concentrates on feasible paths for each agent; For Social attention, highlights
              which other agents are most important to focus on when predicting the trajectory of the agent i)
        """

        # Feature 1 processing

        # print("Type feature_1: ", type(feature_1))
        # print("Type feature_decoder: ", type(feature_decoder))

        if (len(feature_1.size()) == 4):
            # Visual Extractor
            feature_1 = feature_1.contiguous().view(-1,feature_1.size(2)*feature_1.size(3)) # 4D -> 2D
        elif (len(feature_1.size()) == 3):
            # Joint Extractor
            feature_1 = feature_1.contiguous().view(-1,feature_1.size(2)) # 3D -> 2D
        # print("\nFeature 1: ", feature_1.shape)
        linear_feature1_output = self.linear_feature(feature_1)

        # Feature decoder processing

        #print("Feature decoder: ", feature_decoder.shape)
        feature_decoder = feature_decoder.contiguous().view(-1,feature_decoder.size(2)) # 3D -> 2D
        #print("Feature decoder 1: ", feature_decoder.shape)
        linear_decoder_output = self.linear_decoder(feature_decoder)
        # print("Feature decoder 2: ", linear_decoder_output.shape)

        # print("\nLinear decoder output: ", linear_decoder_output.shape)
        # print("Linear feature1 output: ", linear_feature1_output.shape)
    
        alpha = self.softmax(linear_decoder_output)
        # print("\nAlpha: ", alpha.shape)
        context_vector = torch.matmul(alpha, linear_feature1_output)
        return alpha, context_vector
