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

    def forward(self, feature_1, feature_decoder, num_agents=32):
        """
        Inputs:
            - feature_1: Visual Extracto8r output (4D {batch*1, 512, 18, 18}) or Joint Extractor output (3D {L, 32*batch, dim_features})
            - feature_decoder: Generator output (LSTM based decoder; 3D {1, num_agents*batch, dim_features})
        Outputs:
            - alpha?
            - context_vector: (For physical attention, concentrates on feasible paths for each agent; For Social attention, highlights
              which other agents are most important to focus on when predicting the trajectory of the agent i)
        """

        # Feature decode processing

        # feature_decoder = feature_decoder.contiguous().view(-1,feature_decoder.size(2)) # 3D -> 2D
        # print("Feature decoder: ", feature_decoder.shape)
        # Feature 1 processing

        batch = feature_decoder.size(1) # num_agents x batch_size
        batch_size = int(batch/num_agents)

        flag = True

        for i in range(batch_size):
            feature_decoder_ind = feature_decoder[:,num_agents*i:num_agents*(i+1),:] 
            feature_decoder_ind = feature_decoder_ind.contiguous().view(-1,feature_decoder.size(2))
            if (len(feature_1.size()) == 4):
                # Visual Extractor
                if not flag:
                    print("Feature 1 physical: ", feature_1.shape)
                feature_1_ind = torch.unsqueeze(feature_1[i, :, :, :],0)
                feature_1_ind = feature_1_ind.contiguous().view(-1,feature_1_ind.size(2)*feature_1_ind.size(3)) # 4D -> 2D
            elif (len(feature_1.size()) == 3):
                # Joint Extractor
                if not flag:
                    print("Feature 1 social: ", feature_1.shape)
                feature_1_ind = feature_1[:,num_agents*i:num_agents*(i+1),:]
                feature_1_ind = feature_1_ind.contiguous().view(-1, num_agents) # 3D -> 2D
                # feature_1_ind = feature_1_ind.contiguous().view(batch, -1) # 3D -> 2D

            # print("feature_1_ind: ", feature_1_ind.shape)
            # print("Linear feature: ", self.linear_feature)
            linear_feature1_output = self.linear_feature(feature_1_ind)
            
            # Feature decoder processing
            linear_decoder_output = self.linear_decoder(feature_decoder_ind)
        
            alpha = self.softmax(linear_decoder_output) # 32 x 512

            if not flag:
                # print("feature 1 ind: ", feature_1_ind.shape)
                # print("linear_feature1_output: ", linear_feature1_output.shape)
                # print("alpha: ", alpha.shape)
                flag = True

            if i == 0:
                list_context_vector = torch.matmul(alpha, linear_feature1_output)
            else:
                context_vector = torch.matmul(alpha, linear_feature1_output)
                list_context_vector = torch.cat((list_context_vector,context_vector), 0)
        
        # print("List context: ", list_context_vector.shape)
        return alpha, list_context_vector
