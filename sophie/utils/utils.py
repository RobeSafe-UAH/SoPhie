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

def relative_to_abs_sgan(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def create_weights(batch, vmin, vmax, w_len=30, w_type="linear"):
    w = torch.ones(w_len)
    if w_type == "linear":
        w = torch.linspace(vmin, vmax, w_len)
    elif w_type == "exponential":
        w = w

    w = torch.repeat_interleave(w.unsqueeze(0), batch, dim=0)
    return w