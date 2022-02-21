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
import pdb
import glob

from pathlib import Path
from prodict import Prodict
from torch.utils.data import DataLoader

from argoverse.evaluation.competition_util import generate_forecasting_h5

sys.path.append("/home/robesafe/tesis/SoPhie")

from sophie.data_loader.argoverse.dataset_sgan_version import ArgoverseMotionForecastingDataset, seq_collate
from sophie.models.sophie_adaptation import TrajectoryGenerator
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.utils import relative_to_abs, relative_to_abs_sgan

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_path', default='data/datasets/argoverse/', type=str)
parser.add_argument('--num_samples', default=6, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--results_path', default='results/argoverse/exp5', type=str)
parser.add_argument('--results_file', default='test_predictions', type=str)

# Global variables

seqs_without_agent = 0

# Evaluate model functions

def evaluate(loader, generator, num_samples, results_path, results_file, pred_len, split):
    """
    """

    total_traj = 0

    output_all = {}
    test_folder = "data/datasets/argoverse/motion-forecasting/test/data/"
    file_list = glob.glob(os.path.join(test_folder, "*.csv"))
    file_list = [int(name.split("/")[-1].split(".")[0]) for name in file_list]
    with torch.no_grad(): # When testing, gradient calculation is not required
        for batch_index, batch in enumerate(loader):
            print(f"Evaluating batch {batch_index+1}/{len(loader)}")

            batch = [tensor.cuda() for tensor in batch] # Use GPU

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
                loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, num_seq_list) = batch

            total_traj += pred_traj_gt.size(1)

            predicted_traj = []
            for _ in range(num_samples):

                # Get predictions
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, frames)

                # Get predictions in absolute coordinates
                pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1])
                agent_idx = int(torch.where(object_cls==1)[0].cpu().item())
                predicted_traj.append(pred_traj_fake[:,agent_idx,:])
            predicted_traj = torch.stack(predicted_traj, axis=0)

            key = num_seq_list[0].cpu().item()
            output_all[key] = predicted_traj.cpu().numpy()
            file_list.remove(key)

        # add sequences not loaded in dataset
        print("file_list ", file_list)
        for key in file_list:
            output_all[key] = np.zeros((num_samples, 30, 2))

        # Generate H5 file for Argoverse Motion-Forecasting competition
        generate_forecasting_h5(output_all, results_path)

    return output_all

# Ad-hoc generator for each dataset

def get_generator(checkpoint, config):
    """
    """
    generator = TrajectoryGenerator(config.sophie.generator)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda() # Use GPU
    generator.eval()
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
    print("Load test split...")
    data_test = ArgoverseMotionForecastingDataset(dataset_name=config_file.dataset_name,
                                                  root_folder=config_file.dataset.path,
                                                  obs_len=config_file.hyperparameters.obs_len,
                                                  pred_len=0,
                                                  distance_threshold=config_file.hyperparameters.distance_threshold,
                                                  split="test",
                                                  num_agents_per_obs=config_file.hyperparameters.num_agents_per_obs,
                                                  split_percentage=1)

    test_loader = DataLoader(data_test,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              collate_fn=seq_collate)

    # Get generator
    print("Load generator...")
    checkpoint = torch.load(args.model_path)
    generator = get_generator(checkpoint.config_cp, config_file)

    # Evaluate, store results in .json file and get metrics

    ## Create results folder if does not exist

    if not os.path.exists(args.results_path):
        print("Create results path folder: ", args.results_path)
        os.mkdir(args.results_path)

    output_all = evaluate(test_loader, generator, args.num_samples, args.results_path, 
                        args.results_file, config_file.hyperparameters.pred_len, 
                        config_file.dataset.split)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)