import yaml
import torch
import numpy as np
from numpy.random import default_rng
from pathlib import Path
from prodict import Prodict
import csv
import pdb
import sys
import os

from sklearn import linear_model
from skimage.measure import LineModelND, ransac

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BASE_DIR = "/home/robesafe/libraries/SoPhie"
sys.path.append(BASE_DIR)

from sophie.utils.utils import relative_to_abs_sgan
# from sophie.models.sophie_adaptation import TrajectoryGenerator
from sophie.models.mp_sovi import TrajectoryGenerator
from sophie.data_loader.argoverse.dataset_sgan_version_test_map import ArgoverseMotionForecastingDataset, \
                                                                       seq_collate, load_list_from_folder, \
                                                                       read_file, process_window_sequence
import sophie.data_loader.argoverse.map_utils as map_utils
from sophie.trainers.trainer_sophie_adaptation import cal_ade, cal_fde
from argoverse.map_representation.map_api import ArgoverseMap
import sophie.data_loader.argoverse.map_utils as map_utils
import sophie.data_loader.argoverse.dataset_utils as dataset_utils

avm = ArgoverseMap()

with open(r'./configs/sophie_argoverse.yml') as config:
            config = yaml.safe_load(config)
            config = Prodict.from_dict(config)
            config.base_dir = BASE_DIR

def evaluate_feasible_area_prediction(pred_traj_fake, filename):
    """
    Get feasible_area_loss. If a prediction point (in pixel coordinates) is in the drivable (feasible)
    area, is weighted with 1. Otherwise, it is weighted with 0. Theoretically, all points must be
    in the prediction area for the AGENT in Argoverse

    Input:
        pred_traj_fake: Torch.tensor -> pred_len x batch_size x 2 (x|y)
        filename: Image filename to read
    Output:
        feasible_area_loss: min = 0 (num_points · 1), max = pred_len (num_points · 1)
    """


    return feasible_area_loss

config.dataset.split = "train"
data_images_folder = config.dataset.path + config.dataset.split + "/data_images"