import torch
from torch import nn

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
        return (
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda()
        )

    def forward(self,input_data):
        """
        Inputs:
        - input_data: Tensor of shape (input_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.hidden_dim)
        """
        batch = input_data.size(1)
        input_embedding = self.spatial_embedding(
            input_data.contiguous().view(-1,2)
        )
        input_embedding = input_embedding.view(
            -1, batch, self.emb_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(input_embedding, state_tuple)
        final_h = state[0]
        return output, final_h
