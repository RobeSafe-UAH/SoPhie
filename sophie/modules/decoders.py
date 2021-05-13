import torch
from torch import nn

from sophie.modules.layers import MLP

class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.num_layers = config.decoder.num_layers
        self.hidden_dim = config.decoder.hidden_dim
        self.emb_dim = config.decoder.emb_dim
        self.decoder_dropout = config.decoder.dropout

        self.spatial_embedding = MLP(config.decoder.mlp)
        self.decoder = nn.LSTM(
            self.emb_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=self.decoder_dropout
        )

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda()
        )

    def forward(self,input_data):
        """
        Define => modificar
        """
        batch = input_data.size(1)
        input_embedding = self.spatial_embedding(
            input_data.contiguous().view(-1,2)
        )
        input_embedding = input_embedding.view(
            -1, batch, self.emb_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.decoder(input_embedding, state_tuple)
        final_h = state[0]
        return output, final_h
