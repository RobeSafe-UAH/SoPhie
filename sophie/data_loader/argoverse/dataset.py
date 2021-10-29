#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 12 18:29:36 2021
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

import psutil
import copy
import cv2
import glob, glob2
import logging 
import math
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch 
import pandas as pd
import time
import gc # Garbage Collector

from argoverse.map_representation.map_api import ArgoverseMap
import sophie.data_loader.argoverse.map_utils as map_utils
from torch.utils.data import Dataset

# Global variables

avm = ArgoverseMap()

def safe_list(input_data):
    """
    """
    safe_data = copy.copy(input_data)
    return safe_data

def safe_path(input_path):
    """
    """
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

def isstring(string_test):
    """
    """
    return isinstance(string_test, str)

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None):
    """
    """
    folder_path = safe_path(folder_path)
    if isstring(ext_filter): ext_filter = [ext_filter]

    full_list = []
    if depth is None: # Find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = os.path.join(wildcard_prefix,'*'+ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path,wildcard))
                if sort: curlist = sorted(curlist)
                full_list += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            full_list += curlist
    else: # Find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                full_list += curlist
            # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort: curlist = sorted(curlist)
            full_list += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            full_list += newlist

    full_list = [os.path.normpath(path_tmp) for path_tmp in full_list]
    num_elem = len(full_list)

    return full_list, num_elem

def distance_filter(sequences, seq_separators, dist_threshold):
  """
  """
  print("Sequences before distance filtering: ", sequences.shape)
  start = time.time()
  filtered_sequences = []
  new_seq_separators = []
  new_seq_separators.append(0)
  aux_seq_separator = 0

  for i in range(len(seq_separators)):
    if i < len(seq_separators)-1:
          sequence = sequences[seq_separators[i]:seq_separators[i+1],:]
    else:
        sequence = sequences[seq_separators[i]:,:]

    ref_x, ref_y = 0, 0

    sequence_distance = []
    av_indices = np.where(sequence[:,1] == 0)[0]

    for j in range(len(av_indices)):
      if j < len(av_indices)-1:
          sub_sequence = sequence[av_indices[j]:av_indices[j+1],:]
      else:
          sub_sequence = sequence[av_indices[j]:,:]
      av_x, av_y = sub_sequence[0,2], sub_sequence[0,3]
      sub_sequence_distances = [math.sqrt(pow(x-av_x,2)+pow(y-av_y,2)) for _,_,x,y,_ in sub_sequence]
      sequence_distance.append(sub_sequence_distances)
    sequence_distance = np.concatenate(sequence_distance).reshape(-1,1)
    to_keep = np.where(sequence_distance[:,0] <= dist_threshold)[0]
    filtered_sequence = np.take(sequence, to_keep, axis=0)
    filtered_sequences.append(filtered_sequence)

    if i < len(seq_separators)-1:
        new_seq_separator = filtered_sequence.shape[0]
        aux_seq_separator += new_seq_separator
        new_seq_separators.append(aux_seq_separator)

  filtered_sequences = np.concatenate(filtered_sequences)
  new_seq_separators = np.array(new_seq_separators)
  print("Sequences after distance filtering: ", filtered_sequences.shape)
  end = time.time()
  print(f"Time consumed by distance filter: {end-start}")
  return filtered_sequences, new_seq_separators

def dummies_filter(filtered_by_distance_sequences, new_seq_separators, num_agents_per_obs=10):
  """
  """
  start = time.time()
  seq_separator_pre = 0
  dummy_filtered_sequences = []

  for t in range(new_seq_separators.shape[0]):
      if t < new_seq_separators.shape[0] - 1:
          filtered_by_distance_sequence = filtered_by_distance_sequences[new_seq_separators[t,0]:new_seq_separators[t+1,0],:]
      else:
          filtered_by_distance_sequence = filtered_by_distance_sequences[new_seq_separators[t,0]:,:]

      obs_windows = np.where(filtered_by_distance_sequence[:,1] == 0)[0]

      for i in range(len(obs_windows)):
          if i < len(obs_windows) - 1:
              agents_in_obs = obs_windows[i+1] - obs_windows[i] - 1 # Not including the ego-vehicle
              sub_sequence = filtered_by_distance_sequence[obs_windows[i]:obs_windows[i+1],:] # Including ego-vehicle
          else:
              agents_in_obs = filtered_by_distance_sequence.shape[0] - obs_windows[i] - 1
              sub_sequence = filtered_by_distance_sequence[obs_windows[i]:,:] # Including ego-vehicle

          if agents_in_obs < num_agents_per_obs - 1: # We introduce dummy data
              timestamp = sub_sequence[0,0]
              dummy_agents = num_agents_per_obs - agents_in_obs - 1
              dummy_array = np.array([timestamp,-1,-1.0,-1.0,-1]) # timestamp, object_id, x, y, object_class
              dummy_array = np.tile(dummy_array,dummy_agents).reshape(-1,5)
              dummy_sub_sequence = np.concatenate([sub_sequence,dummy_array])
          elif agents_in_obs == num_agents_per_obs - 1:
              dummy_sub_sequence = sub_sequence
          else:
              agents_dist = []

              for j in range(sub_sequence.shape[0]):
                  if sub_sequence[j,1] == 0: # The AV starts the observations of a sequence and also for each timestamp
                      av_x = sub_sequence[j,2]
                      av_y = sub_sequence[j,3]
                  else:
                      obj_x = sub_sequence[j,2]
                      obj_y = sub_sequence[j,3]

                      dist = math.sqrt(pow(obj_x-av_x,2)+pow(obj_y-av_y,2))
                      agents_dist.append(dist)
              agents_dist = np.array(agents_dist)
              sorted_indeces = np.argsort(agents_dist)

              to_delete_indeces = sorted_indeces[num_agents_per_obs-1:] # Only keep the closest num_agents_per_obs agents 
                                                                    # (-1 since index starts in 0)
              dummy_sub_sequence = np.delete(sub_sequence,to_delete_indeces,axis=0)

          dummy_filtered_sequences.append(dummy_sub_sequence)

  dummy_filtered_sequences = np.concatenate(dummy_filtered_sequences)
  print("Dummy filtered sequences: ", dummy_filtered_sequences.shape)
  end = time.time()
  print(f"Time consumed by dummy filter: {end-start}")
  return dummy_filtered_sequences  

def relative_displacements(num_sequences, fixed_sized_sequences, num_agents_per_obs=10, num_obs=50, num_last_obs=19):
    """
    """
    start = time.time()
    num_positions = num_obs * num_agents_per_obs
    print("Num sequences: ", num_sequences)
    print("Sequences: ", fixed_sized_sequences.shape)
    print("Num positions: ", num_positions)
    assert num_sequences == fixed_sized_sequences.shape[0] / num_positions
    print("Num sequences: ", num_sequences)
    
    relative_sequences = []
    ego_vehicle_origin = []

    for i in range(num_sequences):
        if i < num_sequences - 1:
            sequence = fixed_sized_sequences[num_positions*i:num_positions*(i+1),:]
        else:
            sequence = fixed_sized_sequences[num_positions*i:,:]

        origin_x = sequence[num_last_obs*num_agents_per_obs,2]
        origin_y = sequence[num_last_obs*num_agents_per_obs,3]
        origin = np.array([origin_x, origin_y]).reshape(1,2)
        ego_vehicle_origin.append(origin)

        for j in range(sequence.shape[0]):
            if np.int64(sequence[j,1]) != -1:
                sequence[j,2] -= origin_x
                sequence[j,3] -= origin_y
        relative_sequences.append(sequence)
    ego_vehicle_origin = np.concatenate(ego_vehicle_origin)
    relative_sequences = np.concatenate(relative_sequences)
    end = time.time()
    print(f"Time consumed by relative displacements: {end-start}")
    
    return relative_sequences, ego_vehicle_origin

def poly_fit(traj, traj_len, threshold):
    """
    Input:
        - traj: Numpy array of shape (2, traj_len)
        - traj_len: Len of trajectory
        - threshold: Minimum error to be considered for non linear traj
    Output:
        - 1.0 -> Non Linear 
        - 0.0 -> Linear
    """ 
    t = np.linspace(0, traj_len-1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]

    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def load_images():
    """
    """


def load_images(video_path, frames, extension="png", new_shape=(600,600)):
    frames_list = []
    cont = 0
    for frame in frames:
        folder_name = get_folder_name(video_path[0], frame[0].item())
        cont += 1
        image_id = str(int(frame[1].item()))
        image_url = os.path.join(folder_name, "{}.{}".format(image_id.zfill(6), extension))
        #print("image_url: ", image_url)
        frame = cv2.imread(image_url)
        frame = cv2.resize(frame, new_shape)
        frames_list.append(np.expand_dims(frame, axis=0))
    frames_arr = np.concatenate(frames_list, axis=0)
    return frames_arr

def seq_collate(data):
    """
    This functions takes as input the dataset output (see __getitem__ function) and transforms to
    a particular format to feed the Pytorch standard dataloader

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt,
    non_linear_obj, loss_mask, seq_id_list, frames_path, 
    frames_extension, frames_list, object_class_id_list, object_id_list)

                                 |
                                 v

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    non_linear_obj, loss_mask, id_frame, frames, "object_cls", 
    "seq", "obj_id") = batch
    """

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt,
     non_linear_obj, loss_mask, seq_id_list, 
     object_class_id_list, object_id_list) = zip(*data)

    batch_size = obs_traj.shape[0]

    _len = [len(seq) for seq in obs_traj]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size

    obs_traj = torch.cat(obs_traj, dim=0).permute(2, 0, 1)
    pred_traj_gt = torch.cat(pred_traj_gt, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=0).permute(2, 0, 1)
    pred_traj_rel_gt = torch.cat(pred_traj_rel_gt, dim=0).permute(2, 0, 1)
    non_linear_obj = torch.cat(non_linear_obj)
    loss_mask = torch.cat(loss_mask, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    id_frame = torch.cat(seq_id_list, dim=0).permute(2, 0, 1) # seq_len - peds_in_curr_seq - 3
    frames = load_images(list(frames_path), list(frames_list), frames_extension[0])
    frames = torch.from_numpy(frames).type(torch.float32)
    frames = frames.permute(0, 3, 1, 2)
    object_cls = torch.stack(object_class_id_list)
    seq = torch.stack(frames_list)
    obj_id = torch.stack(object_id_list)

    out = [
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, non_linear_obj,
            loss_mask, seq_start_end, frames, object_cls, seq, obj_id
          ] # id_frame??????

    return tuple(out)

num_objs_in_seq = []
seq_list = []
seq_list_rel = []
loss_mask_list = []
non_linear_obj = []
seq_id_list = []
frames_list = []
object_class_id_list = []
object_id_list = []
curr_seq_data = []
relative_sequences = []

class ArgoverseMotionForecastingDataset(Dataset):

    def __init__(self, root_dir, trajectory_file, sequence_separators_file, obs_len=20, pred_len=30, skip=1, threshold=0.002,
                 min_objs=0, windows_frames=None, phase='train', delim='\t', num_agents=10):
        """
        - root_dir: Directory containing the main files 
        - trajectory_file: File (.npy) with all sequences vertically concatenated in the format <frame_id> <object_id> <x> <y>
        - sequence_separators_file: File (.npy) with the indexes to separate the original sequences (before filtering). We must use
          separators since each trajectory has a different number of lines (observations)
        - obs_len: Number of observed frames in prior (input) trajectories
        - pred_len: Number of predicted frames in posterior (output) trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using 
          a linear predictor
        - min_objs: Minimum number of objects that should be in a sequence
        - windows_frames: Specific frames to analize the past and predict the future from that point
        - phase: train (GT is provided) or test (GT not provided)
        - delim: Delimiter in the dataset files
        - num_agents: Number of agents to be considered in a single forward (including the ego-vehicle). If there are less than num_agents,
          dummy variables are used to predict. If there are more, agents are filtered by its distance to the ego-vehicle
          forwards but using the same physical information and frame index
        """

        # Initialize variables

        super(ArgoverseMotionForecastingDataset, self).__init__()
        self.root_dir = root_dir
        self.dataset_name = "argoverse_motion_forecasting_dataset"
        self.objects_id_dict = {"Car": 0, "Cyc": 1, "Mot": 2, "Ped": 3} # TODO: Get this by argument
        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.skip, self.delim = skip, delim
        self.num_agents = num_agents

        # num_objs_in_seq = []
        # seq_list = []
        # seq_list_rel = []
        # loss_mask_list = []
        # non_linear_obj = []
        # seq_id_list = []
        # frames_list = []
        # object_class_id_list = []
        # object_id_list = []

        self.repo_folder = "/home/robesafe/libraries/SoPhie/"
        self.parent_folder = "data/datasets/argoverse/motion-forecasting/train/"

        # Load files and prepare the trajectories

        folder = self.repo_folder + self.parent_folder + "data"
        files, num_files = load_list_from_folder(folder)
        file_id_list = []
        for file_name in files:
            file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
            file_id_list.append(file_id)
        file_id_list.sort()

        with open(sequence_separators_file, 'rb') as seq_sep_file:
          seq_separators = np.load(seq_sep_file).reshape(-1)
          print("Separators shape: ", seq_separators.shape)

        # Load trajectory_file and sequence_separators
        """
        with open(trajectory_file, 'rb') as obs_file:
          joined_obs_trajectories =  np.load(obs_file)
          print("Trajectories shape: ", joined_obs_trajectories.shape)

        with open(sequence_separators_file, 'rb') as seq_sep_file:
          seq_separators = np.load(seq_sep_file).reshape(-1)
          print("Separators shape: ", seq_separators.shape)

        # Get maximum distance between AV and AGENT for the last observation of the past positions

        max_distance = 0
        index_max_distance = 0
        distances = np.zeros((seq_separators.shape[0],1))

        for i in range(len(seq_separators)):
          # print("i: ", i)
          start = seq_separators[i]
          if i < len(seq_separators) - 1:  
            end = seq_separators[i+1]
            sequence = joined_obs_trajectories[start:end,:]
          else:
            sequence = joined_obs_trajectories[start:,:]

          av_indeces = np.where(sequence[:,1] == 0)[0]
          agent_indeces = np.where(sequence[:,1] == 1)[0]
          av_last_observation_row = av_indeces[obs_len-1]
          agent_last_observation_row = agent_indeces[obs_len-1]

          x_av, y_av = sequence[av_last_observation_row,2], sequence[av_last_observation_row,3]
          x_agent, y_agent = sequence[agent_last_observation_row,2], sequence[agent_last_observation_row,3]

          dist = math.sqrt(pow(x_av-x_agent,2)+pow(y_av-y_agent,2))
          distances[i] = dist
          
          if dist > max_distance:
            max_distance = dist
            index_max_distance = i

        print("Max distance: ", max_distance)
        print("Index max distance: ", index_max_distance)
        print("Mean: ", distances.mean())
        print("Min: ", distances.min())
        print("Max: ", distances.max())
        over_100 = np.where(distances[:,0]>50)[0].shape
        print("over 50: ", over_100)
        
        # Apply distance filter based on maximum distance between agent and ego-vehicle
          
        print("-----------------> Distance filter")
        distance_filtered_sequences, new_seq_separators = distance_filter(joined_obs_trajectories, seq_separators, max_distance)
        del joined_obs_trajectories # Free memory
        gc.collect()

        distance_filtered_sequences_file = self.repo_folder + self.parent_folder + "distance_filtered_sequences.npy"
        print("Save: ", distance_filtered_sequences.shape)
        with open(distance_filtered_sequences_file, 'wb') as aux_file:
            np.save(aux_file, distance_filtered_sequences)

        new_seq_separators_file = self.repo_folder + self.parent_folder + "new_seq_separators.npy"
        print("Save: ", new_seq_separators)
        with open(new_seq_separators_file, 'wb') as aux_file:
            np.save(aux_file, new_seq_separators)    

        # Apply dummy filter to get a fixed shape of each sequence: {Num_obs (50 in this case) x Num_max_agents (10, for example, including tge ego-vehicle)
        # x 5 (columns), so it results in a tensor of 500 x 5 x num_sequences (num_files)

        distance_filtered_sequences_file = self.repo_folder + self.parent_folder + "distance_filtered_sequences.npy"
        with open(distance_filtered_sequences_file, 'rb') as aux_file:
            distance_filtered_sequences = np.load(aux_file).reshape(-1,5)
            print("Shape distance filtered: ", distance_filtered_sequences.shape)

        new_seq_separators_file = self.repo_folder + self.parent_folder + "new_seq_separators.npy"
        with open(new_seq_separators_file, 'rb') as aux_file:
            new_seq_separators = np.load(aux_file).reshape(-1,1)
            print("New seq shape: ", new_seq_separators.shape)

        print("-----------------> Dummy filter")
        fixed_sized_sequences = dummies_filter(distance_filtered_sequences, new_seq_separators, num_agents_per_obs=10)
        del distance_filtered_sequences # Free memory
        gc.collect()


        fixed_size_sequences_file = self.repo_folder + self.parent_folder + "fixed_size_sequences.npy"
        print("Save: ", fixed_sized_sequences.shape)
        with open(fixed_size_sequences_file, 'wb') as aux_file:
            np.save(aux_file, fixed_sized_sequences)
        
        # Get relative displacements

        fixed_size_sequences_file = self.repo_folder + self.parent_folder + "fixed_size_sequences.npy"
        with open(fixed_size_sequences_file, 'rb') as aux_file:
            fixed_sized_sequences = np.load(aux_file).reshape(-1,5)

        print("-----------------> Relative displacements")
        num_sequences = 205942
        relative_sequences, ego_vehicle_origin = relative_displacements(num_sequences, fixed_sized_sequences, num_agents_per_obs=10, num_obs=50, num_last_obs=19)
        del fixed_sized_sequences # Free memory
        gc.collect()

        # Save checkpoint

        relative_sequences_file = self.repo_folder + self.parent_folder + "relative_sequences.npy"
        with open(relative_sequences_file, 'wb') as aux_file:
            np.save(aux_file, relative_sequences)

        ego_vehicle_origin_file = self.repo_folder + self.parent_folder + "ego_vehicle_origin.npy"
        with open(ego_vehicle_origin_file, 'wb') as aux_file:
            np.save(aux_file, ego_vehicle_origin)
        """

        # ram_pre = psutil.virtual_memory().percent
        # print("RAM pre: ", ram_pre)
        
        relative_sequences_file = self.repo_folder + self.parent_folder + "relative_sequences.npy"
        # relative_sequences = np.array([])
        with open(relative_sequences_file, 'rb') as aux_file:
            relative_sequences = np.load(aux_file)
        print("Relative sequences: ", relative_sequences.shape)

        # ram_mid = psutil.virtual_memory().percent
        # print("RAM mid: ", ram_mid)

        ego_vehicle_origin_file = self.repo_folder + self.parent_folder + "ego_vehicle_origin.npy"
        with open(ego_vehicle_origin_file, 'rb') as aux_file:
            ego_vehicle_origin = np.load(aux_file).reshape(-1,2)
        print("Ego vehicle origin: ", ego_vehicle_origin.shape)

        # del relative_sequences
        # # indexes_del = tuple(np.arange(relative_sequences.shape[0]).tolist())
        # # relative_sequences = np.delete(relative_sequences, indexes_del, 0)
        # # print("Shape: ", relative_sequences.shape)
        # gc.collect()

        # for _ in range(100):
        #     ram_post = psutil.virtual_memory().percent
        #     print("RAM post: ", ram_post)
        # ram_post = psutil.virtual_memory().percent
        # print("RAM post: ", ram_post)

        # assert 1 == 0

        # print("Relative sequences: ", relative_sequences[:100,:], relative_sequences.shape)

        num_positions = num_agents * self.seq_len
        num_sequences = seq_separators.shape[0] 
        print("Num positions: ", num_positions)
        print("Num sequences: ", num_sequences)

        start = time.time()

        # Rasterized map

        self.city_id = np.load(self.root_dir+"/city_id.npy")
        self.ego_origin = np.load(self.root_dir+"/ego_vehicle_origin.npy")
        self.dist_rasterized_map = [-40,40,-40,40]

        # Main for -> Load whole dataset

        # for seq_index in range(seq_separators.shape[0]):
        for seq_index in range(10):
            print("seq_index: ", seq_index)
            if seq_index < num_sequences - 1:
                curr_seq_data = relative_sequences[num_positions*seq_index:num_positions*(seq_index+1),:] # Frame - ID - X - Y - Class
            else:
                curr_seq_data = relative_sequences[num_positions*(seq_index):] # Frame - ID - X - Y - Class

            

            # Get the corresponding rasterized map

            # rasterized_start = time.time()

            # curr_city = city_id[seq_index]
            # if curr_city == 0:
            #     city_name = "PIT"
            # else:
            #     city_name = "MIA"

            # curr_ego_origin = ego_origin[seq_index].reshape(1,-1)
            # start = time.time()
            # fig = map_utils.map_generator(curr_seq_data, curr_ego_origin, dist_rasterized_map, self.avm, city_name, show=False, smoothen=True)
            # end = time.time()
            # # print(f"Time consumed by map generator: {end-start}")
            # start = time.time()
            # img = map_utils.renderize_image(fig)
            # plt.close("all")
            # end = time.time()
            # # print(f"Time consumed by map render: {end-start}")

            # rasterized_end = time.time()
            # print(f"Time consumed by rasterized image: {rasterized_end-rasterized_start}")
            
            
            continue

            curr_seq_timestamps = curr_seq_data[:,0]
            curr_seq_timestamps = curr_seq_timestamps[::self.num_agents] # Take the timestamp per obs (each obs has self.num_agents)
            continue
            # From relatives to global coordinates

            # print("Relatives: ", curr_seq_data[:10,:])
            curr_seq_data[:,2:4] += ego_vehicle_origin[seq_index,:]
            # print("Global: ", curr_seq_data[:10,:])

            # curr_seq_data dimensions: Rows (= Num_agents_per_obs x Num_obs) x Columns (5)

            # Initialize data

            continue

            curr_seq_rel   = np.zeros((self.num_agents, 2, self.seq_len)) # num_agents x 2 x seq_len
            curr_seq       = np.zeros((self.num_agents, 2, self.seq_len)) # num_agents x 2 x seq_len
            curr_loss_mask = np.zeros((self.num_agents,    self.seq_len)) # num_agents x seq_len
            id_frame_list  = np.zeros((self.num_agents, 3, self.seq_len)) # num_agents x 3 x seq_len
            object_class_list = np.zeros(self.num_agents)
            object_class_list.fill(-1) # By default, -1 (dummy agents)
            id_frame_list.fill(0)
            num_objs_considered = 0
            _non_linear_obj = []

            objs_in_curr_seq = np.unique(curr_seq_data[:,1]) # Number of objects in the window

            # Loop through every object in this window

            num_dummies = 0

            for obj_index, obj_id in enumerate(objs_in_curr_seq):
                if obj_id == -1:
                    num_objs_considered += 1
                    continue
                elif obj_index > self.num_agents - 1:
                    continue

                # if obj_id == -1 or num_objs_considered > self.num_agents - 1:
                #     continue

                # print("Obj index, Obj id: ", obj_index, obj_id)
                object_indexes = np.where(curr_seq_data[:,1]==obj_id)[0]
                num_obs_obj_id = len(object_indexes)

                object_id_index = object_indexes[0] # First index in which this object appears
                object_class_id = curr_seq_data[object_id_index,4]

                curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :] # Frame - ID - X - Y - Class  for one of the id of the window, same id
                
                # Only keep the state

                cache_tmp = np.transpose(curr_obj_seq[:,:2]) # 2 x seq_len -> First row = [0,:] list of frames | Second row = [1,:] id
                curr_obj_seq = np.transpose(curr_obj_seq[:,2:-1]) # 2 x seq_len -> First row = [0,:] x | Second row = [1,:] y
                                                                  # Global coordinates
                
                if num_obs_obj_id < self.seq_len:
                    object_indexes = np.array([])
                    num_dummies += 1
                    id_dummy = -1 * np.ones((1,self.seq_len))
                    cache_tmp = np.vstack([curr_seq_timestamps,id_dummy])

                    x_dummy = np.random.rand(1,50)
                    y_dummy = np.random.rand(1,50)
                    curr_obj_seq = np.vstack([x_dummy,y_dummy])
                                                                  
                # Make coordinates relative

                rel_curr_obj_seq = np.zeros(curr_obj_seq.shape)
                rel_curr_obj_seq[:,1:] = curr_obj_seq[:,1:] - curr_obj_seq[:,:-1]
                _idx = num_objs_considered
                curr_seq[_idx, :, :] = curr_obj_seq
                curr_seq_rel[_idx, :, :] = rel_curr_obj_seq

                # Record seqname, frame and ID information 

                id_frame_list[_idx, :2, :] = cache_tmp
                id_frame_list[_idx,  2, :] = file_id_list[seq_index] # Not required in Argoverse, we already know the link between trajectories and files

                # Linear vs Non-Linear Trajectory, only fit for the future part, not past part

                if phase != 'test':
                    _non_linear_obj.append(poly_fit(curr_obj_seq, pred_len, threshold))

                # Add mask onto padded dummy data

                object_indexes_obs = (object_indexes / self.num_agents).astype(np.int64)
                curr_loss_mask[_idx, object_indexes_obs] = 1

                # Object ID

                object_class_list[num_objs_considered] = object_class_id
                num_objs_considered += 1

            if num_objs_considered > min_objs:
                if len(_non_linear_obj) != num_objs_considered:
                    dummy = [-1 for i in range(num_objs_considered - len(_non_linear_obj))]
                    _non_linear_obj = _non_linear_obj + dummy
                non_linear_obj += _non_linear_obj
                num_objs_in_seq.append(num_objs_considered)
                loss_mask_list.append(curr_loss_mask[:num_objs_considered])
                seq_list.append(curr_seq[:num_objs_considered])
                seq_list_rel.append(curr_seq_rel[:num_objs_considered])
                seq_id_list.append(id_frame_list[:num_objs_considered])
                # frames_list.append(seq_frame)
                # frames_list.append(img)
                object_class_id_list.append(object_class_list)
                object_id_list.append(id_frame_list[:num_objs_considered][:,1,0])

        end = time.time()
        print(f"Time consumed by sequences preprocessor: {end-start}")

        # self.num_seq = len(seq_list)
        # seq_list = np.concatenate(seq_list, axis=0) # Objects x 2 x seq_len
        # seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        # loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        # non_linear_obj = np.asarray(non_linear_obj)
        # seq_id_list = np.concatenate(seq_id_list, axis=0)
        # # frames_list = np.asarray(frames_list)
        # object_class_id_list = np.asarray(object_class_id_list)
        # object_id_list = np.asarray(object_id_list)

        # Free memory

        ram_pre = psutil.virtual_memory().percent
        print("RAM pre: ", ram_pre)
        # relative_sequences = None
        # del relative_sequences
        # curr_seq_data = None
        # del curr_seq_data
        # print("Dir pre: ", dir())
        # print("Globals pre: ", globals())
        for name in dir():
            if not name.startswith('_') and name != "self":
                # print("Size: ", sys.getsizeof(name))
                # del name
                print("name: ", name)
                try:
                    print("entro: ")
                    del globals()[name]
                    # name = None
                    # del name
                    gc.collect()
                except: 
                    continue

        print("Dir post: ", dir())
        # print("Globals post: ", globals())

        # indexes_del = tuple(np.arange(relative_sequences.shape[0]).tolist())
        # relative_sequences = np.delete(relative_sequences, indexes_del, 0)
        # print("Shape: ", relative_sequences.shape)
        gc.collect()

        # for _ in range(100):
        #     ram_post = psutil.virtual_memory().percent
        #     print("RAM post: ", ram_post)
        ram_post = psutil.virtual_memory().percent
        print("RAM post: ", ram_post)

        print("After concatenate")

        assert 1 == 0

        # Convert numpy -> Torch Tensor

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_gt = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel_gt = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_obj = torch.from_numpy(non_linear_obj).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_objs_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.seq_id_list = torch.from_numpy(seq_id_list).type(torch.float)
        # self.frames_list = torch.from_numpy(frames_list).type(torch.float)
        self.object_class_id_list = torch.from_numpy(object_class_id_list).type(torch.float)
        self.object_id_list = torch.from_numpy(object_id_list).type(torch.float)

        print("After torch")

        processing_folder = self.repo_folder + self.parent_folder + "after_dataset_processing"
        if not os.path.isdir(processing_folder):
            os.mkdir(processing_folder)

        torch.save(self.obs_traj, processing_folder+"/obs_traj.pt")
        torch.save(self.pred_traj_gt, processing_folder+"/pred_traj_gt.pt")
        torch.save(self.obs_traj_rel, processing_folder+"/obs_traj_rel.pt")
        torch.save(self.pred_traj_rel_gt, processing_folder+"/pred_traj_rel_gt.pt")
        torch.save(self.loss_mask, processing_folder+"/loss_mask.pt")
        torch.save(self.non_linear_obj, processing_folder+"/non_linear_obj.pt")
        torch.save(self.seq_start_end, processing_folder+"/seq_start_end.pt")
        torch.save(self.seq_id_list, processing_folder+"/seq_id_list.pt")
        # torch.save(self.frames_list, processing_folder+"/frames_list.pt")
        torch.save(self.object_class_id_list, processing_folder+"/object_class_id_list.pt")
        torch.save(self.object_id_list, processing_folder+"/object_id_list.pt")

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
                self.obs_traj[start:end, :], self.pred_traj_gt[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel_gt[start:end, :],
                self.non_linear_obj[start:end], self.loss_mask[start:end, :],
                self.seq_id_list[start:end, :], self.object_class_id_list[index], 
                self.object_id_list[index], self.city_id[index], self.ego_origin[index], self.dist_rasterized_map
              ]   

        return out

