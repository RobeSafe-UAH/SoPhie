import torch
from torch import nn
from prodict import Prodict
from sophie.modules.layers import MLP

class Encoder(nn.Module):

    def __init__(self, num_layers, hidden_dim, emb_dim, mlp_config, dropout=0.4):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.encoder_dropout = dropout
        self.spatial_embedding = MLP(**mlp_config) # nn.Linear(x, emb_dim_mlp)
        self.encoder = nn.LSTM(
            self.emb_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=self.encoder_dropout
        )

    def init_hidden(self, batch):
        if torch.cuda.is_available():
            return (
                torch.zeros(self.num_layers, batch, self.hidden_dim).cuda(), # Hidden state at time 0
                torch.zeros(self.num_layers, batch, self.hidden_dim).cuda()  # Cell state at time 0
            )

    def forward(self,input_data):
        """
        Inputs:
        - input_data: Tensor of shape (input_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.hidden_dim)
        """

        #print("spatial_embedding: ", self.spatial_embedding)
        #print("<<<<< Input data shape: ", input_data.shape)
        batch = input_data.size(1)
        dim_features_size = input_data.size(2)
        #print("Self emb: ", self.emb_dim)

        test = input_data.contiguous().view(-1,dim_features_size)
        #print("Test: ", test.shape)
        input_embedding = self.spatial_embedding(
            input_data.contiguous().view(-1,dim_features_size)
        )
        #print("Input embedding: ", input_embedding.shape)

        #print("> ", input_embedding.shape)
        input_embedding = input_embedding.view(
            -1, batch, self.emb_dim
        )
        #print(">> ", input_embedding.shape)

        if torch.cuda.is_available():
            input_embedding = input_embedding.float()
            input_embedding = input_embedding.cuda()

        state_tuple = self.init_hidden(batch)

        output, states = self.encoder(input_embedding, state_tuple)
        #print(">>> ", output.shape)

        # print("Output: ", output, output.shape, type(output))
        # print("Output: ", type(output))
        # print("Hidden states: ", states[0], states[0].shape)
        # print("Cell states: ", states[1], states[1].shape)

        final_h = states[0]
        
        return output, final_h