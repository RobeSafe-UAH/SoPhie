import torch
from torch import nn

from sophie.modules.layers import MLP

class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.emb_dim = config.emb_dim
        self.decoder_dropout = config.dropout
        self.pred_len = config.pred_len
        self.predicted_trajectories_generator = MLP(**config.mlp_config) # nn.Linear(x, emb_dim_mlp)

        self.decoder = nn.LSTM(
            self.emb_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=self.decoder_dropout
        )

        self.spatial_embedding = nn.Linear(
            config.linear_1.input_dim,
            config.linear_1.output_dim
        )
        self.hidden2pos = nn.Linear(
            config.linear_2.input_dim,
            config.linear_2.output_dim
        )

    def init_hidden(self, num_agents=32):
        if torch.cuda.is_available():
            return (
                torch.zeros(self.num_layers, num_agents, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, num_agents, self.hidden_dim).cuda()
            )

            # return (
            #     torch.zeros(self.num_layers, num_agents, self.hidden_dim),
            #     torch.zeros(self.num_layers, num_agents, self.hidden_dim)
            # )

    def forward(self, input_data, num_agents=32):
        """
        input_data: (batch_size*num_agents, batch_size*num_agents)
        """

        batch = input_data.size(0)
        batch_size = int(batch/num_agents)
        state_tuple = self.init_hidden(num_agents)

        for i in range(batch_size):
            predicted_trajectories = []

            input_data_ind = input_data[num_agents*i:num_agents*(i+1),num_agents*i:num_agents*(i+1)]
            input_embedding = self.spatial_embedding(input_data_ind) # 32 -> 64
            input_embedding = input_embedding.view(
                1, num_agents, self.emb_dim
            )

            for _ in range(self.pred_len):
                output, state = self.decoder(input_embedding, state_tuple)
                embedding_input = self.hidden2pos(output.view(-1,self.hidden_dim))
                aux = embedding_input.view(-1, self.emb_dim)
                rel_pos = self.predicted_trajectories_generator(aux)
                input_embedding = embedding_input.view(1, num_agents, self.emb_dim)
                predicted_trajectories.append(rel_pos.view(num_agents, -1))

            pred_traj_fake_rel = torch.stack(predicted_trajectories, dim=0)

            if i == 0:
                list_pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len,pred_traj_fake_rel.shape[1],-1)
            else:
                pred_traj_fake_rel = pred_traj_fake_rel.view(self.pred_len,pred_traj_fake_rel.shape[1],-1)
                list_pred_traj_fake_rel = torch.cat((list_pred_traj_fake_rel,pred_traj_fake_rel), 1)
        
        return list_pred_traj_fake_rel, state_tuple[0] 
