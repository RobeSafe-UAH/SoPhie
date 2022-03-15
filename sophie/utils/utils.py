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


def relative_to_abs_sgan_multimodal(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (b, m, t, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    displacement = torch.cumsum(rel_traj, dim=2)
    start_pos = torch.unsqueeze(torch.unsqueeze(start_pos, dim=1), dim=1)
    abs_traj = displacement + start_pos
    return abs_traj

def relative_to_abs(rel_traj, start_pos):
    # TODO: Not used in the training stage. Nevertheless, rewrite using torch, not numpy
    """
    Inputs:
    - rel_traj: numpy array of shape (len, 2)
    - start_pos: numpy array of shape (1, 2)
    Outputs:
    - abs_traj: numpy array of shape (len, 2) in absolute coordinates
    """

    displacement = np.cumsum(rel_traj, axis=0)
    abs_traj = displacement + start_pos

    return abs_traj

def create_weights(batch, vmin, vmax, w_len=30, w_type="linear"):
    w = torch.ones(w_len)
    if w_type == "linear":
        w = torch.linspace(vmin, vmax, w_len)
    elif w_type == "exponential":
        w = w

    w = torch.repeat_interleave(w.unsqueeze(0), batch, dim=0)
    return w

def freeze_model(model, no_freeze_list=[]):
    for name, child in model.named_children():
        if name not in no_freeze_list:
            for param in child.parameters():
                param.requires_grad = False

    return model