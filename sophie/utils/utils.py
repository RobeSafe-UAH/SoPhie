import torch
import copy
import pdb

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch * num_agents_per_obs, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch * num_agents_per_obs, 2)
    """

    # rel_traj = rel_traj.permute(1, 0, 2)
    # displacement = torch.cumsum(rel_traj, dim=1)
    # start_pos = torch.unsqueeze(start_pos, dim=1)
    # pdb.set_trace()
    # abs_traj = displacement + start_pos
    # return abs_traj.permute(1, 0, 2)

    # print("Relative trajectories: ", rel_traj, rel_traj.shape)
    # print("Start pos: ", start_pos, start_pos.shape)

    batch_size = start_pos.shape[0]
    num_agents_per_obs = int(rel_traj.shape[1] / batch_size)
    # print("batch_size, agents: ", batch_size, num_agents_per_obs)

    abs_traj = rel_traj.clone()

    for seq_index in range(batch_size):
        with torch.no_grad():
            if seq_index < batch_size - 1:
                abs_traj[:,seq_index*num_agents_per_obs:(seq_index+1)*num_agents_per_obs,:] += start_pos[seq_index,:]
            else:
                abs_traj[:,seq_index*num_agents_per_obs:,:] += start_pos[seq_index,:]

    # print("Absolute trajectories: ", abs_traj, abs_traj.shape)

    return abs_traj