import torch
from torch import nn

from sophie.modules.layers import MLP

class Classifier(nn.Module):

    def __init__(self, mlp_config):
        super(Classifier, self).__init__()

        self.spatial_embedding = MLP(**mlp_config) 
        self.softmax = nn.Softmax(dim=1) # Perform softmax operation accross hidden_dim axis
    """
    # Sotfmax: https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    def forward(self, trajectory_predicted, trajectory_groundtruth):
        embedding = [trajectory_predicted,trajectory_groundtruth]
        return self.softmax(embedding) # Return binary classification: Fake (Predicted) vs True (Groundtruth) trajectories
    """

    def forward(self, input_data):
        """
        Inputs:
        - input_data: Tensor of shape (self.num_layers (LSTM encoder discriminator), batch, self.hidden_dim)
        Output:
        - predicted_labels: Predicted label from the discriminator (Batch size x 2)
        """

        batch_size = input_data.size(1)
        
        input_embedding = self.spatial_embedding(
            input_data.contiguous().view(batch_size,-1)
        )

        predicted_labels = self.softmax(input_embedding) 

        return predicted_labels