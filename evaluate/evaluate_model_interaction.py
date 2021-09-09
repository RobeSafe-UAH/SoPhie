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

from sophie.data_loader.interaction.dataset import read_file, seq_collate_image, InteractionDataset
from sophie.models import SoPhieGenerator
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.utils import relative_to_abs

from evaluate import evaluate_aux_functions

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_path', default='data/datasets/aiodrive/aiodrive_Car/', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--results_path', default='results/aiodrive', type=str)
parser.add_argument('--results_file', default='test_json', type=str)
parser.add_argument('--skip', default=1, type=int)

def get_generator(checkpoint, config): # Standard or dataset-specific ?????????????
    config.sophie.generator.decoder.linear_3.input_dim = config.dataset.batch_size*2*config.sophie.generator.social_attention.linear_decoder.out_features
    config.sophie.generator.decoder.linear_3.output_dim = config.dataset.batch_size*config.number_agents
    generator = SoPhieGenerator(config.sophie.generator)
    generator.build()
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def evaluate(loader, generator, num_samples, pred_len, results_path, results_file, test_submission=False):
    init_json = False
    json_dict = AutoTree()

    final_ade, final_fde = 0,0
    ade_outer, fde_outer = [], []
    total_traj = 0

    with torch.no_grad():
        

def main(args):
    if os.path.isdir(args.model_path): # Model path is a folder
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, file_) for file_ in filenames]
    else:
        paths = [args.model_path] # Model_path is a file

    BASE_DIR = Path(__file__).resolve().parent.parent

    with open(r'/home/robesafe/libraries/SoPhie/configs/sophie_interaction.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR

    test_submission = config_file.dataset.test_submission

    # windows_frames, skip, obs_len ... ?
    obs_len = config_file.hyperparameters.obs_len
    pred_len = config_file.hyperparameters.pred_len

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint.config_cp, config_file)
        test_path = os.path.join(config_file.base_dir, args.dataset_path, "test") # Trajectories
        videos_path = os.path.join(config_file.base_dir, config_file.dataset.video) # Images

        # Dataset

        data_test = InteractionDataset(test_path=test_path, videos_path=videos_path, 
                                       obs_len=obs_len, pred_len=pred_len, phase='testing')

        # Dataloader

        test_loader = DataLoader(data_test, 
                                 batch_size=config_file.dataset.batch_size,
                                 shuffle=config_file.dataset.shuffle,
                                 num_workers=config_file.dataset.num_workers,
                                 collate_fn=seq_collate_image
                                 )

        # Evaluate, store results and get metrics

        ade, fde = evaluate(checkpoint.config_cp, test_loader,
                            generator, args.num_samples, pred_len,
                            args.results_path, args.results_file,
                            test_submission)

        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
               args.dataset_path, pred_len, ade, fde))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)