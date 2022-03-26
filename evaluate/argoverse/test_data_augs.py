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

# Fill some additional dimensions

past_observations = config.hyperparameters.obs_len
num_agents_per_obs = config.hyperparameters.num_agents_per_obs
config.sophie.generator.social_attention.linear_decoder.out_features = past_observations * num_agents_per_obs

config.dataset.split = "val"
config.dataset.split_percentage = 0.0002 # To generate the final results, must be 1 (whole split test)
config.dataset.start_from_percentage = 0.0
config.dataset.batch_size = 1 # Better to build the h5 results file
config.dataset.num_workers = 0
config.dataset.class_balance = -1.0 # Do not consider class balance in the split val
config.dataset.shuffle = False

config.hyperparameters.pred_len = 30 # In test, we do not have the gt (prediction points) -> 0

data_images_folder = config.dataset.path + config.dataset.split + "/data_images"
data_images_augs_folder = config.dataset.path + config.dataset.split + "/data_images_augs"

seq_len = config.hyperparameters.obs_len + config.hyperparameters.pred_len
threshold = 0.002

dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]

if not os.path.exists(data_images_augs_folder):
    print("Create experiment path: ", data_images_augs_folder)
    os.mkdir(data_images_augs_folder)

folder = config.dataset.path + config.dataset.split + "/data/"
files, num_files = load_list_from_folder(folder)

file_id_list = []
root_file_name = None
for file_name in files:
    if not root_file_name:
        root_file_name = os.path.dirname(os.path.abspath(file_name))
    file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
    file_id_list.append(file_id)
file_id_list.sort()
print("Num files: ", num_files)

if config.dataset.shuffle:
    rng = default_rng()
    indeces = rng.choice(num_files, size=int(num_files*config.dataset.split_percentage), replace=False)
    file_id_list = np.take(file_id_list, indeces, axis=0)
else:
    start_from = int(config.dataset.start_from_percentage*num_files)
    n_files = int(config.dataset.split_percentage*num_files)
    file_id_list = file_id_list[start_from:start_from+n_files]

    if (start_from + n_files) >= num_files:
        file_id_list = file_id_list[start_from:]
    else:
        file_id_list = file_id_list[start_from:start_from+n_files]

check_data_augs = [0,0,0] # swapping, dropout (erasing), gaussian noise
check_rotations = False # rotations

for i, file_id in enumerate(file_id_list):
    path = os.path.join(root_file_name,str(file_id)+".csv")
    data = read_file(path) 

    frames = np.unique(data[:, 0]).tolist() 
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :]) # save info for each frame

    idx = 0
    rot_angle = -1

    num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, \
    curr_seq_rel, id_frame_list, object_class_list, city_id, ego_origin = \
                process_window_sequence(idx, frame_data, frames, \
                                        seq_len, config.hyperparameters.pred_len, threshold, 
                                        file_id, config.dataset.split, config.hyperparameters.obs_origin,
                                        rot_angle=rot_angle,augs=check_data_augs)

    curr_seq = torch.from_numpy(curr_seq).permute(2, 0, 1)
    curr_seq_rel = torch.from_numpy(curr_seq_rel).permute(2, 0, 1)

    first_obs = curr_seq[0,:,:]

    curr_city = round(city_id)
    if curr_city == 0:
        city_name = "PIT"
    else:
        city_name = "MIA"

    filename = data_images_folder + "/" + str(file_id) + ".png"

    img = map_utils.plot_trajectories(filename, curr_seq_rel, first_obs, 
                                        ego_origin, object_class_list, dist_rasterized_map,
                                        rot_angle=rot_angle,obs_len=config.hyperparameters.obs_len, 
                                        smoothen=False, show=True)

    plt.close("all")

    if check_rotations: # check rotations
        rots_angles = [0,90,180,270]

        for rot_angle in rots_angles:
            num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, \
            curr_seq_rel, id_frame_list, object_class_list, city_id, ego_origin = \
                process_window_sequence(idx, frame_data, frames, \
                                        seq_len, config.hyperparameters.pred_len, threshold, 
                                        file_id, config.dataset.split, config.hyperparameters.obs_origin,
                                        rot_angle=rot_angle,augs=None)

            curr_seq = torch.from_numpy(curr_seq).permute(2, 0, 1)
            curr_seq_rel = torch.from_numpy(curr_seq_rel).permute(2, 0, 1)

            first_obs = curr_seq[0,:,:]

            curr_city = round(city_id)
            if curr_city == 0:
                city_name = "PIT"
            else:
                city_name = "MIA"

            filename = data_images_folder + "/" + str(file_id) + ".png"

            img = map_utils.plot_trajectories(filename, curr_seq_rel, first_obs, 
                                                ego_origin, object_class_list, dist_rasterized_map,
                                                rot_angle=rot_angle,obs_len=config.hyperparameters.obs_len, 
                                                smoothen=True, show=True)

            plt.close("all")

    if i == 9:
        break


