#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""

Created on Thu Sep 09 13:04:56 2021
@author: Carlos Gómez-Huélamo

"""

import argparse
import json
import numpy as np
import os
import sys
import torch
import yaml

from pathlib import Path
from prodict import Prodict
from torch.utils.data import DataLoader

sys.path.append("/home/robesafe/libraries/SoPhie")

from sophie.data_loader.argoverse.dataset import ArgoverseMotionForecastingDataset
# from sophie.data_loader.argoverse.dataset import read_file, seq_collate_image, ArgoverseDataset
from sophie.models import SoPhieGenerator
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.utils import relative_to_abs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_path', default='data/datasets/argoverse/', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--results_path', default='results/aiodrive', type=str)
parser.add_argument('--results_file', default='test_json', type=str)
parser.add_argument('--skip', default=1, type=int)

# Auxiliar functions

class AutoTree(dict):
    """
    Dictionary with unlimited levels
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def condition(x,class_name):
    """
    """
    return x==class_name

def evaluate_helper(error, seq_start_end):
    """
    """
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

# Ad-hoc generator for each dataset

def get_generator(checkpoint, config):
    """
    """
    config.sophie.generator.decoder.linear_3.input_dim = config.dataset.batch_size*2*config.sophie.generator.social_attention.linear_decoder.out_features
    config.sophie.generator.decoder.linear_3.output_dim = config.dataset.batch_size*config.number_agents
    generator = SoPhieGenerator(config.sophie.generator)
    generator.build()
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

# Evaluate model functions

def store_json(json_dict, predicted_trajectories_og, object_cls_og, seq_og,obj_id_og, prediction_length):
    """
    """
    # TODO: Get multimodal prediction, not only the best one (At this moment, it is the same both for '0' and '1')
    # TODO: Store pred_fake_trajectories in Argoverse format
    trajectory_samples = ['0','1']
    prob = 1.0 # Does not make sense, if multimodal, each prediction should have a certain probability
    na = 32 # Number of agents
    batch_size = int(predicted_trajectories_og.shape[1]/na)

    # print("keys 1: ", json_dict['10'].keys())

    for element in range(batch_size):
        object_cls = object_cls_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 32 -> 1 x 32
        seq = seq_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 2 -> 1 x 2
        obj_id = obj_id_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 32 -> 1 x 32 
        pred_fake_trajectories = predicted_trajectories_og[:,na*element:na*(element+1),:] # 12 x batch_size*na (8x32) x 2 -> 12 x 32 x 2
        
        object_indexes = []

        # AIODRIVE format
        # xx = str(int(seq[0,0]/1000))
        # yyyy = str(int(seq[0,0]%1000))

        # seq_name = ""

        # if xx == '10':
        #     seq_name = "Town"+xx.zfill(2)+"HD"+"_seq"+yyyy.zfill(4)
        # else:
        #     seq_name = "Town"+xx.zfill(2)+"_seq"+yyyy.zfill(4)

        # seq_frame = str(int(seq[0,1]))

        # for key,value in classes.items():
        #     if value == -1: # Dummy class
        #         continue
        #     indexes = np.where(np.array([condition(xi,value) for xi in object_cls]))[1]

        #     agent_dict = {}
        #     # print("Indexes: ", indexes)
        #     for i in range(pred_fake_trajectories.shape[1]): 
        #         if i in indexes:
        #             ground_pos = []
        #             for j in range(pred_fake_trajectories.shape[0]): 
        #                 if j<10:
        #                     ground_pos.append(pred_fake_trajectories[j,i,:].tolist()) # x,y
        #             aux_dict = {}
        #             aux_dict['state'] = ground_pos
        #             aux_dict['prob'] = prob
        #             agent_dict[str(int(obj_id[0,i]))] = aux_dict

        #     keys = json_dict[str(prediction_length)].keys()
        #     if key in keys:
        #         if json_dict[str(prediction_length)][key][seq_name][seq_frame][trajectory_samples[0]]: # Not empty
        #             previous_agent_dict = json_dict[str(prediction_length)][key][seq_name][seq_frame][trajectory_samples[0]]
        #             previous_agent_dict.update(agent_dict)
        #             agent_dict = previous_agent_dict
        #     if agent_dict: # Not empty
        #         for trajectory_sample in trajectory_samples:
        #             json_dict[str(prediction_length)][key][seq_name][seq_frame][trajectory_sample] = agent_dict
    return json_dict

def evaluate(loader, generator, num_samples, pred_len, results_path, results_file, test_submission=False, skip=1):
    """
    """
    init_json = False
    json_dict = AutoTree()
    prediction_length = 10*skip # 10 (1 s), 20 (2 s), 50 (5 s)

    final_ade, final_fde = 0,0
    ade_outer, fde_outer = [], []
    total_traj = 0

    with torch.no_grad(): # When testing, gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            print("Evaluating batch: ", batch_index)
            batch = [tensor.cuda() for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_obj, loss_mask, frames, object_class, 
             seq_frame, object_id) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(frames, obs_traj)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1]) # Absolute coordinates

                json_dict = store_json(json_dict, pred_traj_fake, object_class, seq_frame, object_id, prediction_length)

                if not test_submission: # We cannot compute ADE and FDE metrics if we do not have the gt data for those frames
                    ade.append(displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    ))

            if not test_submission:
                ade_sum = evaluate_helper(ade, seq_start_end)
                fde_sum = evaluate_helper(fde, seq_start_end)

                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)

        if not test_submission:
            final_ade = sum(ade_outer) / (total_traj * pred_len)
            final_fde = sum(fde_outer) / (total_traj)
        else:
            final_ade = -1
            final_fde = -1        

def main(args):
    root_dir = "../data/datasets/argoverse/motion-forecasting/train"
    trajectory_file = "../data/datasets/argoverse/motion-forecasting/train/joined_obs_trajectories.npy"
    sequence_separators_file = "../data/datasets/argoverse/motion-forecasting/train/sequence_separators.npy"
    data_test = ArgoverseMotionForecastingDataset(root_dir = root_dir,
                                                  trajectory_file = trajectory_file,
                                                  sequence_separators_file = sequence_separators_file)


    # Dataloader

    # test_loader = DataLoader(data_test, 
    #                          batch_size=config_file.dataset.batch_size,
    #                          shuffle=config_file.dataset.shuffle,
    #                          num_workers=config_file.dataset.num_workers,
    #                          collate_fn=seq_collate_image
    #                          )

    # # Evaluate, store results in .json file and get metrics

    # ade, fde = evaluate(checkpoint.config_cp, test_loader,
    #                     generator, args.num_samples, pred_len,
    #                     args.results_path, args.results_file,
    #                     test_submission)

    # print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
    #        args.dataset_path, pred_len, ade, fde))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)