import argparse
import gc
import logging
import os
import sys
import time
from prodict import Prodict
import pdb
import numpy as np
import random
from torchviz import make_dot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from sophie.data_loader.argoverse.dataset_unified import ArgoverseMotionForecastingDataset, seq_collate
from sophie.data_loader.argoverse.dataset_sgan_version import ArgoverseMotionForecastingDataset, seq_collate
from sophie.models.sophie_adaptation import TrajectoryGenerator
from sophie.models.sophie_adaptation import TrajectoryDiscriminator
from sophie.modules.losses import gan_g_loss, gan_d_loss, l2_loss, gan_d_loss_bce, gan_g_loss_bce
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.checkpoint_data import Checkpoint, get_total_norm
from sophie.utils.utils import relative_to_abs, relative_to_abs_sgan

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes(use_gpu):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def handle_batch(batch, is_single_agent_out):
    # load batch in cuda
    batch = [tensor.cuda() for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list) = batch
    
    # handle single agent 
    agent_idx = None
    if is_single_agent_out: # search agent idx
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()
        pred_traj_gt = pred_traj_gt[:,agent_idx, :]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]

    return (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list)

def n_data(data, vmin, vmax):
    return (data - vmin) / (vmax - vmin)

def dn_data(data, vmin, vmax):
    return data * (vmax - vmin) + vmin

def model_trainer(config, logger):
    """
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    long_dtype, float_dtype = get_dtypes(config.use_gpu)

    logger.info('Configuration: ')
    logger.info(config)

    logger.info("Initializing train dataset") 
    data_train = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                   root_folder=config.dataset.path,
                                                   obs_len=config.hyperparameters.obs_len,
                                                   pred_len=config.hyperparameters.pred_len,
                                                   distance_threshold=config.hyperparameters.distance_threshold,
                                                   split="train",
                                                   num_agents_per_obs=config.hyperparameters.num_agents_per_obs,
                                                   split_percentage=config.dataset.split_percentage,
                                                   shuffle=config.dataset.shuffle,
                                                   batch_size=config.dataset.batch_size,
                                                   class_balance=config.dataset.class_balance,
                                                   obs_origin=config.hyperparameters.obs_origin)

    train_loader = DataLoader(data_train,
                              batch_size=config.dataset.batch_size,
                              shuffle=config.dataset.shuffle,
                              num_workers=config.dataset.num_workers,
                              collate_fn=seq_collate)

    logger.info("Initializing val dataset")
    data_val = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                 root_folder=config.dataset.path,
                                                 obs_len=config.hyperparameters.obs_len,
                                                 pred_len=config.hyperparameters.pred_len,
                                                 distance_threshold=config.hyperparameters.distance_threshold,
                                                 split="val",
                                                 num_agents_per_obs=config.hyperparameters.num_agents_per_obs,
                                                 split_percentage=config.dataset.split_percentage,
                                                 shuffle=config.dataset.shuffle,
                                                 class_balance=config.dataset.class_balance,
                                                 obs_origin=config.hyperparameters.obs_origin)
    val_loader = DataLoader(data_val,
                            batch_size=config.dataset.batch_size,
                            shuffle=config.dataset.shuffle,
                            num_workers=config.dataset.num_workers,
                            collate_fn=seq_collate)

    tn = next(iter(train_loader))
    vn = next(iter(val_loader))
    # select absolute min max norm
    tn_abs = tn[-1][0,0]
    vn_abs = vn[-1][0,0]
    abs_norm = (min(tn_abs[0], vn_abs[0]), max(tn_abs[1], vn_abs[1]))

    # select relative min max norm
    tn_rel = tn[-1][0,1]
    vn_rel = vn[-1][0,1]
    rel_norm = (min(tn_rel[0], vn_rel[0]), max(tn_rel[1], vn_rel[1]))

    hyperparameters = config.hyperparameters
    optim_parameters = config.optim_parameters
    if not hyperparameters.classic_trainer:
        iterations_per_epoch = len(data_train) / config.dataset.batch_size / hyperparameters.d_steps
        if hyperparameters.num_epochs:
            hyperparameters.num_iterations = int(iterations_per_epoch * hyperparameters.num_epochs)
            hyperparameters.num_iterations = hyperparameters.num_iterations if hyperparameters.num_iterations != 0 else 1
    else:
        iterations_per_epoch = len(data_train) / config.dataset.batch_size
        # select stop condition: epoch or iterations
        if (hyperparameters.num_iterations > hyperparameters.num_epochs* iterations_per_epoch) and (hyperparameters.num_epochs != 0):
            hyperparameters.num_iterations = hyperparameters.num_epochs* iterations_per_epoch

    hyperparameters["abs_norm"] = abs_norm
    hyperparameters["rel_norm"] = rel_norm
    logger.info("abs_norm: {}".format(abs_norm))
    logger.info("abs_norm: {}".format(rel_norm))

    logger.info(
        'There are {} iterations per epoch'.format(hyperparameters.num_iterations)
    )

    generator = TrajectoryGenerator()
    generator.to(device)
    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Generator model:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator()
    discriminator.to(device)
    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Discriminator model:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss_bce
    d_loss_fn = gan_d_loss_bce
    optimizer_g = optim.Adam(generator.parameters(), lr=optim_parameters.g_learning_rate, weight_decay=optim_parameters.g_weight_decay)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=optim_parameters.d_learning_rate, weight_decay=optim_parameters.d_weight_decay
    )
    criterion = nn.BCELoss()

    restore_path = None
    if hyperparameters.checkpoint_start_from is not None:
        restore_path = hyperparameters.checkpoint_start_from
    elif hyperparameters.restore_from_checkpoint == 1:
        restore_path = os.path.join(hyperparameters.output_dir,
                                    '%s_with_model.pt' % hyperparameters.checkpoint_name)

    # checkpoint:
    #   g_state, d_state, g_optim_state, d_optim_state, counters { t, epoch }, restore_ts
    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = Checkpoint()
    t0 = None

    if hyperparameters.tensorboard_active:
        exp_path = os.path.join(
            config.base_dir, hyperparameters.output_dir, "tensorboard_logs"
        )
        os.makedirs(exp_path, exist_ok=True)
        writer = SummaryWriter(exp_path)
    logger.info(f"Train {len(train_loader)}")
    logger.info(f"Val {len(val_loader)}")
    while t < hyperparameters.num_iterations:
        gc.collect()
        d_steps_left = hyperparameters.d_steps
        g_steps_left = hyperparameters.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader: # bottleneck

            if not hyperparameters.classic_trainer:
                if d_steps_left > 0:
                    step_type = 'discriminator'

                    losses_d = discriminator_step(hyperparameters, batch, generator,
                                                  discriminator, d_loss_fn,
                                                  optimizer_d, criterion)

                    checkpoint.config_cp["norm_d"].append(
                        get_total_norm(discriminator.parameters()))
                    d_steps_left -= 1
                elif g_steps_left > 0:
                    step_type = 'generator'

                    losses_g = generator_step(hyperparameters, batch, generator,
                                              discriminator, g_loss_fn,
                                              optimizer_g, criterion)
                    checkpoint.config_cp["norm_g"].append(
                        get_total_norm(generator.parameters())
                    )
                    g_steps_left -= 1

                if d_steps_left > 0 or g_steps_left > 0:
                    continue
            else:
                losses_d, losses_g = classic_trainer(hyperparameters, batch, generator, discriminator, g_loss_fn, optimizer_g, optimizer_d, criterion)
                checkpoint.config_cp["norm_d"].append(
                        get_total_norm(discriminator.parameters())
                    )
                checkpoint.config_cp["norm_g"].append(
                        get_total_norm(generator.parameters())
                    )
            if t % hyperparameters.print_every == 0:
                # print logger
                logger.info('t = {} / {}'.format(t + 1, hyperparameters.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["D_losses"].keys():
                        checkpoint.config_cp["D_losses"][k] = []
                    checkpoint.config_cp["D_losses"][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["G_losses"].keys():
                        checkpoint.config_cp["G_losses"][k] = [] 
                    checkpoint.config_cp["G_losses"][k].append(v)
                checkpoint.config_cp["losses_ts"].append(t)

            if t > 0 and t % hyperparameters.checkpoint_every == 0:
                checkpoint.config_cp["counters"]["t"] = t
                checkpoint.config_cp["counters"]["epoch"] = epoch
                checkpoint.config_cp["sample_ts"].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    hyperparameters, val_loader, generator, discriminator, d_loss_fn
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    if hyperparameters.tensorboard_active:
                        writer.add_scalar(k, v, t+1)
                    if k not in checkpoint.config_cp["metrics_val"].keys():
                        checkpoint.config_cp["metrics_val"][k] = []
                    checkpoint.config_cp["metrics_val"][k].append(v)

                min_ade = min(checkpoint.config_cp["metrics_val"]['ade'])
                min_ade_nl = min(checkpoint.config_cp["metrics_val"]['ade_nl'])
                if metrics_val['ade'] <= min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint.config_cp["best_t"] = t
                    checkpoint.config_cp["g_best_state"] = generator.state_dict()
                    checkpoint.config_cp["d_best_state"] = discriminator.state_dict()

                if metrics_val['ade_nl'] <= min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint.config_cp["best_t_nl"] = t
                    checkpoint.config_cp["g_best_nl_state"] = generator.state_dict()
                    checkpoint.config_cp["d_best_nl_state"] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                if metrics_val['ade'] <= min_ade:
                    checkpoint.config_cp["g_state"] = generator.state_dict()
                    checkpoint.config_cp["g_optim_state"] = optimizer_g.state_dict()
                    checkpoint.config_cp["d_state"] = discriminator.state_dict()
                    checkpoint.config_cp["d_optim_state"] = optimizer_d.state_dict()
                    checkpoint_path = os.path.join(
                        config.base_dir, hyperparameters.output_dir, "{}_{}_with_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
                    )
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)
                    logger.info('Done.')

                    # Save a checkpoint with no model weights by making a shallow
                    # copy of the checkpoint excluding some items
                    checkpoint_path = os.path.join(
                        config.base_dir, hyperparameters.output_dir, "{}_{}_no_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
                    )
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    key_blacklist = [
                        'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                        'g_optim_state', 'd_optim_state', 'd_best_state',
                        'd_best_nl_state'
                    ]
                    small_checkpoint = {}
                    for k, v in checkpoint.config_cp.items():
                        if k not in key_blacklist:
                            small_checkpoint[k] = v
                    torch.save(small_checkpoint, checkpoint_path)
                    logger.info('Done.')

            t += 1
            d_steps_left = hyperparameters.d_steps
            g_steps_left = hyperparameters.g_steps
            if t >= hyperparameters.num_iterations:
                break

    ###
    logger.info("Training finished")

    # Check stats on the validation set
    t += 1
    epoch += 1
    checkpoint.config_cp["counters"]["t"] = t
    checkpoint.config_cp["counters"]["epoch"] = epoch+1
    checkpoint.config_cp["sample_ts"].append(t)
    logger.info('Checking stats on val ...')
    metrics_val = check_accuracy(
        hyperparameters, val_loader, generator, discriminator, d_loss_fn
    )

    for k, v in sorted(metrics_val.items()):
        logger.info('  [val] {}: {:.3f}'.format(k, v))
        if hyperparameters.tensorboard_active:
            writer.add_scalar(k, v, t+1)
        if k not in checkpoint.config_cp["metrics_val"].keys():
            checkpoint.config_cp["metrics_val"][k] = []
        checkpoint.config_cp["metrics_val"][k].append(v)

    min_ade = min(checkpoint.config_cp["metrics_val"]['ade'])
    min_ade_nl = min(checkpoint.config_cp["metrics_val"]['ade_nl'])

    if metrics_val['ade'] <= min_ade:
        logger.info('New low for avg_disp_error')
        checkpoint.config_cp["best_t"] = t
        checkpoint.config_cp["g_best_state"] = generator.state_dict()
        checkpoint.config_cp["d_best_state"] = discriminator.state_dict()

    if metrics_val['ade_nl'] <= min_ade_nl:
        logger.info('New low for avg_disp_error_nl')
        checkpoint.config_cp["best_t_nl"] = t
        checkpoint.config_cp["g_best_nl_state"] = generator.state_dict()
        checkpoint.config_cp["d_best_nl_state"] = discriminator.state_dict()

    # Save another checkpoint with model weights and
    # optimizer state
    checkpoint.config_cp["g_state"] = generator.state_dict()
    checkpoint.config_cp["g_optim_state"] = optimizer_g.state_dict()
    checkpoint.config_cp["d_state"] = discriminator.state_dict()
    checkpoint.config_cp["d_optim_state"] = optimizer_d.state_dict()
    checkpoint_path = os.path.join(
        config.base_dir, hyperparameters.output_dir, "{}_{}_with_model.pt".format(config.dataset_name, hyperparameters.checkpoint_name)
    )


def discriminator_step(
    hyperparameters, batch, generator, discriminator, d_loss_fn, optimizer_d, criterion
):
    batch = [tensor.cuda() for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _,_) = batch

    # placeholder loss
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    # single agent output idx
    agent_idx = None
    if hyperparameters.output_single_agent:
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

    # get norm
    abs_norm = (hyperparameters.abs_norm[0].cuda(), hyperparameters.abs_norm[1].cuda())
    rel_norm = (hyperparameters.rel_norm[0].cuda(), hyperparameters.rel_norm[1].cuda())

    # forward
    # generator_out = generator(obs_traj, obs_traj_rel, frames, agent_idx, seq_start_end) # Social Attention per sequence
    generator_out = generator(obs_traj, obs_traj_rel, frames, agent_idx)
    # generator_out = generator(
    #     n_data(obs_traj, abs_norm[0], abs_norm[1]), 
    #     n_data(obs_traj_rel, rel_norm[0], rel_norm[1]), 
    #     frames, 
    #     agent_idx
    # )

    # generator_out = dn_data(generator_out, rel_norm[0], rel_norm[1])

    # single agent trajectories
    if hyperparameters.output_single_agent:
        obs_traj = obs_traj[:,agent_idx, :]
        pred_traj_gt = pred_traj_gt[:,agent_idx, :]
        obs_traj_rel = obs_traj_rel[:, agent_idx, :]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]

    # rel to abs
    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1])

    # calculate full traj
    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(
        traj_fake,
        traj_fake_rel
    )
    scores_real = discriminator(
        traj_real,
        traj_real_rel
    )

    # Compute loss with optional gradient penalty
    # data_loss = d_loss_fn(scores_real, scores_fake, criterion)
    data_loss = gan_d_loss(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()
    D_x = scores_real.mean().item()
    D_G_z1 = scores_fake.mean().item()
    losses["D_x"] = D_x
    losses["D_G_z1"] = D_G_z1

    optimizer_d.zero_grad()
    loss.backward()
    if hyperparameters.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 hyperparameters.clipping_threshold_d)
    optimizer_d.step()

    return losses

def generator_step(
    hyperparameters, batch, generator, discriminator, g_loss_fn, optimizer_g, criterion
):
    batch = [tensor.cuda() for tensor in batch]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
     loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _, _) = batch

    # place holder loss
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []
    # single agent output idx
    agent_idx = None
    if hyperparameters.output_single_agent:
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

    if hyperparameters.output_single_agent:
        loss_mask = loss_mask[agent_idx, hyperparameters.obs_len:]
        pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]
    else:  # 160x30 -> 0 o 1
        loss_mask = loss_mask[:, hyperparameters.obs_len:]

    # get norm
    abs_norm = (hyperparameters.abs_norm[0].cuda(), hyperparameters.abs_norm[1].cuda())
    rel_norm = (hyperparameters.rel_norm[0].cuda(), hyperparameters.rel_norm[1].cuda())

    for _ in range(hyperparameters.best_k):
        # forward
        # generator_out = generator(obs_traj, obs_traj_rel, frames, agent_idx, seq_start_end) # Social Attention per sequence
        generator_out = generator(obs_traj, obs_traj_rel, frames, agent_idx)

        # forward
        # generator_out = generator(
        #     n_data(obs_traj, abs_norm[0], abs_norm[1]), 
        #     n_data(obs_traj_rel, rel_norm[0], rel_norm[1]), 
        #     frames, 
        #     agent_idx
        # )
        # generator_out = dn_data(generator_out, rel_norm[0], rel_norm[1])
        pred_traj_fake_rel = generator_out
        # single agent trajectories # TODO this overwrite full traj
        if hyperparameters.output_single_agent:
            pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1,agent_idx, :])
        else:
            pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1])

        if hyperparameters.l2_loss_weight > 0:
            g_l2_loss_rel.append(hyperparameters.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))
    # handle single agent output
    if hyperparameters.output_single_agent:
        obs_traj = obs_traj[:,agent_idx, :]
        pred_traj_gt = pred_traj_gt[:,agent_idx, :]
        obs_traj_rel = obs_traj_rel[:, agent_idx, :]
    # calculate l2 loss
    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if hyperparameters.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        if hyperparameters.output_single_agent:
            for i in range(len(agent_idx)):
                _g_l2_loss_rel = g_l2_loss_rel[i].unsqueeze(0)
                _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
                _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                    loss_mask[i].unsqueeze(0))
                g_l2_loss_sum_rel += _g_l2_loss_rel
        else:
            for start, end in seq_start_end.data:
                _g_l2_loss_rel = g_l2_loss_rel[start:end]
                _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
                _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                    loss_mask[start:end])
                g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    # calculate full traj
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    # discriminator scores
    scores_fake = discriminator(traj_fake, traj_fake_rel)
    # discriminator_loss = g_loss_fn(scores_fake, criterion)
    discriminator_loss = gan_g_loss(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()
    D_G_z2 = scores_fake.mean().item()
    losses["D_G_z2"] = D_G_z2

    optimizer_g.zero_grad()
    loss.backward()
    if hyperparameters.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), hyperparameters.clipping_threshold_g
        )
    optimizer_g.step()

    return losses

def classic_trainer(hyperparameters, batch, generator, discriminator, g_loss_fn, optimizer_g, optimizer_d, criterion):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
        loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _,_) = batch

    # single agent output idx
    agent_idx = None
    if hyperparameters.output_single_agent:
        agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

    # get norm
    abs_norm = (hyperparameters.abs_norm[0].cuda(), hyperparameters.abs_norm[1].cuda())
    rel_norm = (hyperparameters.rel_norm[0].cuda(), hyperparameters.rel_norm[1].cuda())

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    discriminator.zero_grad()
    
    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    output = discriminator(
        traj_real if agent_idx is None else traj_real[:,agent_idx,:],
        traj_real_rel if agent_idx is None else traj_real_rel[:,agent_idx,:]
    )
    label = torch.ones_like(output) * torch.from_numpy(np.random.uniform(0.8, 1, tuple(output.shape)).astype(np.float32)).cuda()
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch

    # generator_out = generator(obs_traj, obs_traj_rel, frames, agent_idx, seq_start_end) # Social Attention per sequence
    generator_out = generator(obs_traj, obs_traj_rel, frames, agent_idx)
    
    # Generate batch of latent vectors
    # generator_out = generator(
    #     n_data(obs_traj, abs_norm[0], abs_norm[1]),
    #     n_data(obs_traj_rel, rel_norm[0], rel_norm[1]),
    #     frames,
    #     agent_idx
    # )

    # rel to abs
    # generator_out = dn_data(generator_out, rel_norm[0], rel_norm[1])
    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs_sgan(
        pred_traj_fake_rel,
        obs_traj[-1] if agent_idx is None else obs_traj[-1,agent_idx,:]
    )
    traj_fake_rel = torch.cat(
        [
            obs_traj_rel if agent_idx is None else obs_traj_rel[:,agent_idx,:], 
            pred_traj_fake_rel
        ]
        , dim=0
    )
    traj_fake = torch.cat(
        [
            obs_traj if agent_idx is None else obs_traj[:,agent_idx,:], 
            pred_traj_fake
        ]
        , dim=0
    )
    output = discriminator(
        traj_fake.detach(),
        traj_fake_rel.detach()
    )
    label = torch.ones_like(output) * torch.from_numpy(np.random.uniform(0, 0.2, tuple(output.shape)).astype(np.float32)).cuda()
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    # clipping
    if hyperparameters.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 hyperparameters.clipping_threshold_d)
    # Update D
    optimizer_d.step()
    losses_d = {"errD":errD.item(), "D_x":D_x, "D_G_z1":D_G_z1}

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    generator.zero_grad()
    output = discriminator(traj_fake, traj_fake_rel)
    label = torch.ones_like(output) * torch.from_numpy(np.random.uniform(0.8, 1, tuple(output.shape)).astype(np.float32)).cuda()
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    # clipping
    if hyperparameters.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), hyperparameters.clipping_threshold_g
        )
    # Update G
    optimizer_g.step()
    D_G_z2 = output.mean().item()
    losses_g = {"errG": errG.item(), "D_G_z2": D_G_z2}
    return losses_d, losses_g

def check_accuracy(
    hyperparameters, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = [], []
    disp_error, disp_error_l, disp_error_nl = [], [], []
    f_disp_error, f_disp_error_l, f_disp_error_nl = [], [], []
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    # get norm
    abs_norm = (hyperparameters.abs_norm[0].cuda(), hyperparameters.abs_norm[1].cuda())
    rel_norm = (hyperparameters.rel_norm[0].cuda(), hyperparameters.rel_norm[1].cuda())
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _, _) = batch

            # single agent output idx
            agent_idx = None
            if hyperparameters.output_single_agent:
                agent_idx = torch.where(object_cls==1)[0].cpu().numpy()

            # mask and linear
            if not hyperparameters.output_single_agent: # TODO corregir con el nuevo dataset
                mask = np.where(obj_id.cpu() == -1, 0, 1)
                mask = torch.tensor(mask, device=obj_id.device).reshape(-1)
            if hyperparameters.output_single_agent:
                # mask = mask[agent_idx]
                non_linear_obj = non_linear_obj[agent_idx]
                loss_mask = loss_mask[agent_idx, hyperparameters.obs_len:]
                linear_obj = 1 - non_linear_obj
            else:  # 160x30 -> 0 o 1
                loss_mask = loss_mask[:, hyperparameters.obs_len:]
                linear_obj = 1 - non_linear_obj

            # # forward
            # pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, frames, agent_idx, seq_start_end)
            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, frames, agent_idx)
            # forward
            # pred_traj_fake_rel = generator(
            #     n_data(obs_traj, abs_norm[0], abs_norm[1]), 
            #     n_data(obs_traj_rel, rel_norm[0], rel_norm[1]), 
            #     frames, 
            #     agent_idx
            # )
            # pred_traj_fake_rel = dn_data(pred_traj_fake_rel, rel_norm[0], rel_norm[1])

            # single agent trajectories
            if hyperparameters.output_single_agent:
                obs_traj = obs_traj[:,agent_idx, :]
                pred_traj_gt = pred_traj_gt[:,agent_idx, :]
                obs_traj_rel = obs_traj_rel[:, agent_idx, :]
                pred_traj_gt_rel = pred_traj_gt_rel[:, agent_idx, :]
            
            print("pred_traj_fake_rel min ", pred_traj_fake_rel.min(), pred_traj_gt_rel.min())
            print("pred_traj_fake_rel max ", pred_traj_fake_rel.max(), pred_traj_gt_rel.max())

            # rel to abs
            pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1])

            # l2 loss
            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj,
                mask if not hyperparameters.output_single_agent else None
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj,
                mask if not hyperparameters.output_single_agent else None
            )

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_obj).item()
            total_traj_nl += torch.sum(non_linear_obj).item()
            if limit and total_traj >= hyperparameters.num_samples_check:
                break
    # metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * hyperparameters.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * hyperparameters.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * hyperparameters.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics

def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel

def cal_ade(pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, consider_ped)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_obj)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_obj)
    return ade, ade_l, ade_nl

def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_obj, non_linear_obj, consider_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], consider_ped)
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_obj
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_obj
    )
    return fde, fde_l, fde_nl
