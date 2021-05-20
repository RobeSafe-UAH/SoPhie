import argparse
import gc
import logging
import os
import sys
import time
from prodict import Prodict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sophie.data_loader.ethucy.dataset import read_file, EthUcyDataset, seq_collate_image
from sophie.models import SoPhieGenerator
from sophie.models import SoPhieDiscriminator
from sophie.modules.losses import gan_g_loss, gan_d_loss, l2_loss
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.checkpoint_data import Checkpoint, get_total_norm
from sophie.utils.utils import relative_to_abs

torch.backends.cudnn.benchmark = True

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


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


def model_trainer(config):
    ##
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(config.base_dir, config.dataset.path, "train")
    val_path = os.path.join(config.base_dir, config.dataset.path, "val")

    long_dtype, float_dtype = get_dtypes(config.use_gpu)

    #?> cargar dataset eth
    logger.info("Initializing train dataset") 
    data_train = EthUcyDataset(train_path, videos_path=os.path.join(config.base_dir, config.dataset.video))
    train_loader = DataLoader(
        data_train,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.num_workers,
        collate_fn=seq_collate_image)

    logger.info("Initializing val dataset")
    data_val = EthUcyDataset(val_path, videos_path=os.path.join(config.base_dir, config.dataset.video))
    val_loader = DataLoader(
        data_val,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.num_workers,
        collate_fn=seq_collate_image)

    hyperparameters = config.hyperparameters
    iterations_per_epoch = len(data_train) / config.dataset.batch_size / hyperparameters.d_steps
    if hyperparameters.num_epochs:
        hyperparameters.num_iterations = int(iterations_per_epoch * hyperparameters.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    generator = SoPhieGenerator(config.sophie.generator)
    generator.build()
    generator.to(device)
    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Generator model:')
    logger.info(generator)

    discriminator = SoPhieDiscriminator(config.sophie.discriminator)
    discriminator.build()
    discriminator.to(device)
    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Discriminator model:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    #print("======= ", generator.parameters())
    optimizer_g = optim.Adam(generator.parameters(), lr=hyperparameters.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=hyperparameters.d_learning_rate
    )

    # t0 = time.time()
    # t1 = time.time()
    # while(t1 - t0 < 120):
    #     print(t1-t0)
    #     t1 = time.time()
    # assert(1==0), "TSU!"
    # Maybe restore from checkpoint ?> modificar
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
    while t < hyperparameters.num_iterations:
        gc.collect()
        d_steps_left = hyperparameters.d_steps
        g_steps_left = hyperparameters.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if hyperparameters.timing == 1:
                # ?> Waits for all kernels in all streams on a CUDA device to complete.
                torch.cuda.synchronize()
                t1 = time.time()

            if d_steps_left > 0:
                step_type = 'discriminator'
                losses_d = discriminator_step(hyperparameters, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint.norm_d.append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'generator'
                losses_g = generator_step(hyperparameters, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint.norm_g.append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if hyperparameters.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('Model: {} step took {}'.format(step_type, t2 - t1))

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if hyperparameters.timing == 1:
                if t0 is not None:
                    logger.info('Iteration {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            if t % hyperparameters.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, hyperparameters.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint.D_losses[k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint.G_losses[k].append(v)
                checkpoint.losses_ts.append(t)

            if t > 0 and t % hyperparameters.checkpoint_every == 0:
                checkpoint.counters.t = t
                checkpoint.counters.epoch = epoch
                checkpoint.sample_ts.append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    hyperparameters, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    hyperparameters, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint.metrics_val[k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint.metrics_train[k].append(v)

                min_ade = min(checkpoint.metrics_val['ade'])
                min_ade_nl = min(checkpoint.metrics_val['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint.best_t = t
                    checkpoint.g_best_state = generator.state_dict()
                    checkpoint.d_best_state = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint.best_t_nl = t
                    checkpoint.g_best_nl_state = generator.state_dict()
                    checkpoint.d_best_nl_state = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint.g_state = generator.state_dict()
                checkpoint.g_optim_state = optimizer_g.state_dict()
                checkpoint.d_state = discriminator.state_dict()
                checkpoint.d_optim_state = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    config.base_dir, hyperparameters.output_dir, '%s_with_model.pt' % hyperparameters.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    hyperparameters.output_dir, '%s_no_model.pt' % hyperparameters.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = hyperparameters.d_steps
            g_steps_left = hyperparameters.g_steps
            if t >= hyperparameters.num_iterations:
                break


def discriminator_step(
    hyperparameters, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end, frames) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    #generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
    generator_out = generator(frames, obs_traj)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    #scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_fake = discriminator(traj_fake)
    #scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)
    scores_real = discriminator(traj_real)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if hyperparameters.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 hyperparameters.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    hyperparameters, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end, frames) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, hyperparameters.obs_len:]

    for _ in range(hyperparameters.best_k):
        #generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
        generator_out = generator(frames, obs_traj)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if hyperparameters.l2_loss_weight > 0:
            g_l2_loss_rel.append(hyperparameters.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if hyperparameters.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    #scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_fake = discriminator(traj_fake)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if hyperparameters.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), hyperparameters.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(
    hyperparameters, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, frames) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, hyperparameters.obs_len:]

            # pred_traj_fake_rel = generator(
            #     obs_traj, obs_traj_rel, seq_start_end
            # )
            pred_traj_fake_rel = generator(
                frames, obs_traj
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            # scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            # scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            scores_fake = discriminator(traj_fake)
            scores_real = discriminator(traj_real)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

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
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= hyperparameters.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
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


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl

