# Adversarial Loss + L2 loss
import torch
import random
import numpy as np
from torch import Tensor
import pdb

def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.8, 1)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def gan_g_loss_bce(scores_fake, bce):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.8, 1)
    return bce(scores_fake, y_fake)


def gan_d_loss_bce(scores_real, scores_fake, bce):
    y_real = torch.ones_like(scores_real) * random.uniform(0.8, 1)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.2)
    loss_real = bce(scores_real, y_real)
    loss_fake = bce(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def mse_weighted(gt, pred, weights):
    """
        With t = predicted points
        gt: (t, b, 2)
        pred: (t, b, 2)
        weights: (b, t)
    """
    try:
        l2 = gt.permute(1, 0, 2) - pred.permute(1, 0, 2) # b,t,2
        l2 = l2**2 # b, t, 2
        l2 = torch.sum(l2, axis=2) # b, t
        l2 = torch.sqrt(l2) # b, t
        l2 = l2 * weights # b, t
        l2 = torch.mean(l2, axis=1) # b
        l2 = torch.mean(l2) # single value
    except Exception as e:
        print(e)
        pdb.set_trace()
    return l2

def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor,
    pred: Tensor,
    confidences: Tensor,
    avails: Tensor,
    epsilon: float = 1.0e-8,
    is_reduce: bool = True,
) -> Tensor:
    """
    from
    https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence/comments

    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow,
    For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each
                                mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability
                        for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (
        batch_size,
        future_len,
        num_coords,
    ), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size,
        num_modes,
    ), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(
        torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))
    ), "confidences should sum to 1"
    assert avails.shape == (
        batch_size,
        future_len,
    ), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - pred) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        # error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time
        error = torch.log(confidences + epsilon) - 0.5 * torch.sum(error, dim=-1)
    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(
        dim=1, keepdim=True
    )  # error are negative at this point, so max() gives the minimum one
    error = (
        -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True))
        - max_value
    )  # reduce modes
    # print("error", error)
    if is_reduce:
        return torch.mean(error)
    else:
        return error


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """
    from
    https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence/comments

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability
                        for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(
        gt, pred.unsqueeze(1), confidences, avails
    )

def evaluate_feasible_area_prediction(pred_traj_fake_abs, distance_threshold, origin_pos, filename):
    """
    Get feasible_area_loss. If a prediction point (in pixel coordinates) is in the drivable (feasible)
    area, is weighted with 1. Otherwise, it is weighted with 0. Theoretically, all points must be
    in the prediction area for the AGENT in Argoverse

    Input:
        pred_traj_fake_abs: Torch.tensor -> pred_len x 2 (x|y) in global (map) coordinates
        filename: Image filename to read
    Output:
        feasible_area_loss: min = 0 (num_points · 1), max = pred_len (num_points · 1)
    """

    img_map = cv2.imread(filename)
    img_map_gray = cv2.cvtColor(img_map,cv2.COLOR_BGR2GRAY)
    height, width = img_map.shape

    xcenter, ycenter = origin_pos[0][0], origin_pos[0][1]
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    # Transform global point to pixel

    m_x = float(width / (x_max - x_min)) # slope
    m_y = float(height / (y_max - y_min))

    i_x = float(-m_x * x_min) # intercept
    i_y = float(-m_y * y_min)

    feasible_area_loss = []

    for i in range(pred_traj_fake_abs.shape[0]): # pred_len
        pix_x = pt[i,0] * m_x + i_x
        pix_y = pt[i,1] * m_y + i_y

        if img_map[pix_y,pix_x] == 255.0:
            feasible_area_loss.append(1)
        else:
            feasible_area_loss.append(0)

    return feasible_area_loss
