import torch
import copy
import pdb
import numpy as np

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

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: numpy array of shape (len, 2)
    - start_pos: numpy array of shape (1, 2)
    Outputs:
    - abs_traj: numpy array of shape (len, 2) in absolute coordinates
    """

    displacement = np.cumsum(rel_traj, axis=0)
    print("start pos: ", start_pos)
    abs_traj = displacement + start_pos

    return abs_traj