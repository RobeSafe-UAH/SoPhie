#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 09 13:04:56 2021
@author: Miguel Eduardo Ortiz Huamaní and Carlos Gómez-Huélamo
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

from argoverse.evaluation.competition_util import generate_forecasting_h5

sys.path.append("/home/robesafe/libraries/SoPhie")

from sophie.data_loader.argoverse.dataset_unified import ArgoverseMotionForecastingDataset, seq_collate
from sophie.models import SoPhieGenerator
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.utils import relative_to_abs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_path', default='data/datasets/argoverse/', type=str)
parser.add_argument('--num_samples', default=2, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--results_path', default='results/argoverse/exp1', type=str)
parser.add_argument('--results_file', default='test_predictions', type=str)

# Global variables

seqs_without_agent = 0

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

# Evaluate model functions

def store_json(json_dict, predicted_trajectories, object_cls, obj_id, num_seq):
    """
    """
    # TODO: Get multimodal prediction, not only the best one (At this moment, it is the same both for '0' and '1')

    global seqs_without_agent

    # print("Pred Traj: ", predicted_trajectories, predicted_trajectories.shape)
    # print("Obj cls: ", object_cls, object_cls.shape)
    # print("Obj id: ", obj_id, obj_id.shape)
    # print("Num seq: ", num_seq, num_seq.shape)

    batch_size = int(object_cls.shape[0])
    num_agents_per_obs = int(predicted_trajectories.shape[1] / batch_size)

    for i in range(batch_size):
        curr_obj_id = obj_id[i].cpu().data.numpy() # (num_agents_per_obs,)
        curr_num_seq = int(num_seq[i].cpu().data.numpy())
        agent_index = np.where(curr_obj_id == 1.0)[0] # AGENT ID

        if agent_index.size > 0:
            agent_pred_trajectory = predicted_trajectories[:,i*num_agents_per_obs+agent_index,:].reshape(30,2).cpu().data.numpy()
        else: # TODO: Fix this -> What should we do if the agent is further than distance_threshold in the obs_len-th frame?
            seqs_without_agent += 1
            agent_pred_trajectory = np.random.randn(predicted_trajectories.shape[0],2)

        # TODO: THIS IS NOT CORRECT: Provisional multimodal prediction (at this moment we do not provide different predictions)

        multimodal_agent_pred_trajectory = np.array([agent_pred_trajectory,agent_pred_trajectory]) # N (different predictions) x 30 x 2
        multimodal_agent_pred_trajectory = multimodal_agent_pred_trajectory.tolist()
        json_dict[curr_num_seq] = multimodal_agent_pred_trajectory
        # print("agent_pred_trajectory: ", agent_pred_trajectory, agent_pred_trajectory.shape)

    return json_dict

def evaluate(loader, generator, num_samples, results_path, results_file, encoding_dict, pred_len, split):
    """
    """

    final_results = os.path.join(results_path, results_file + '.json')

    json_dict = AutoTree()

    final_ade, final_fde = 0,0
    ade_outer, fde_outer = [], []
    total_traj = 0

    with torch.no_grad(): # When testing, gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            print(f"Evaluating batch {batch_index+1}/{len(loader)}")

            batch = [tensor.cuda() for tensor in batch] # Use GPU
            # batch = [tensor for tensor in batch] # Use CPU

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, ego_vehicle_origin, num_seq) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                # Get predictions

                pred_traj_fake_rel = generator(frames, obs_traj)

                # Get predictions in absolute coordinates

                ego_vehicle_origin = ego_vehicle_origin.reshape(ego_vehicle_origin.shape[0],-1) # batch_size x 1 x 2 -> batch_size x 2
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, ego_vehicle_origin) # Absolute coordinates

                json_dict = store_json(json_dict, pred_traj_fake, object_cls, obj_id, num_seq)

                if split != "test": # We cannot compute ADE and FDE metrics if we do not have the gt data for those frames
                    ade.append(displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    ))

            if split != "test":
                ade_sum = evaluate_helper(ade, seq_start_end)
                fde_sum = evaluate_helper(fde, seq_start_end)

                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)

        if split != "test":
            final_ade = sum(ade_outer) / (total_traj * pred_len)
            final_fde = sum(fde_outer) / (total_traj)
        else:
            final_ade = -1
            final_fde = -1  

        # Store JSON

        # print(("JSON dict: ", json_dict, type(json_dict)))
        with open(final_results, 'w') as output_file:
            json.dump(json_dict, output_file)

        # Generate H5 file for Argoverse Motion-Forecasting competition
        
        generate_forecasting_h5(json_dict, results_path)

    return final_ade, final_fde

# Ad-hoc generator for each dataset

def get_generator(checkpoint, config):
    """
    """

    generator = SoPhieGenerator(config.sophie.generator)
    generator.set_num_agents(config.hyperparameters.num_agents_per_obs)
    generator.build()
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda() # Use GPU
    generator.train()
    return generator      

def main(args):
    """
    """

    # Load config file

    BASE_DIR = Path(__file__).resolve().parent

    print("BASE_DIR: ", BASE_DIR)

    with open(r'./configs/sophie_argoverse.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        print(yaml.dump(config_file, default_flow_style=False))
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR

    ## Fill some additional dimensions

    past_observations = config_file.hyperparameters.obs_len
    num_agents_per_obs = config_file.hyperparameters.num_agents_per_obs
    config_file.sophie.generator.social_attention.linear_decoder.out_features = past_observations * num_agents_per_obs

    # Dataloader

    data_test = ArgoverseMotionForecastingDataset(dataset_name=config_file.dataset_name,
                                                  root_folder=config_file.dataset.path,
                                                  obs_len=config_file.hyperparameters.obs_len,
                                                  pred_len=config_file.hyperparameters.pred_len,
                                                  distance_threshold=config_file.hyperparameters.distance_threshold,
                                                  split=config_file.dataset.split,
                                                  num_agents_per_obs=config_file.hyperparameters.num_agents_per_obs,
                                                  split_percentage=config_file.dataset.split_percentage)

    test_loader = DataLoader(data_test,
                              batch_size=config_file.dataset.batch_size,
                              shuffle=config_file.dataset.shuffle,
                              num_workers=config_file.dataset.num_workers,
                              collate_fn=seq_collate)

    # Get generator

    checkpoint = torch.load(args.model_path)
    generator = get_generator(checkpoint.config_cp, config_file)

    # Load encoding ids dict

    encoding_ids_file = os.path.join(config_file.dataset.path,config_file.dataset.split,"ids_file.json")
    print("encoding_ids_file: ", encoding_ids_file)
    with open(encoding_ids_file) as input_file:
        encoding_dict = json.load(input_file)

    # Evaluate, store results in .json file and get metrics

    ## Create results folder if does not exist

    if not os.path.exists(args.results_path):
        print("Create results path folder: ", args.results_path)
        os.mkdir(args.results_path)

    ade, fde = evaluate(test_loader, generator, args.num_samples, args.results_path, 
                        args.results_file, encoding_dict, config_file.hyperparameters.pred_len, 
                        config_file.dataset.split)

    print('\n\nDataset: {}, Pred Len: {:.2f} s, ADE: {:.2f}, FDE: {:.2f}'.format(args.dataset_path, 
                                                                             config_file.hyperparameters.pred_len/10, 
                                                                             ade, fde))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)