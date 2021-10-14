#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 12 18:29:36 2021
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

import copy
import cv2
import glob, glob2
import logging 
import math
import numpy as np
import os
import sys
import torch 
import pandas as pd
import time
import gc # Garbage Collector

from torch.utils.data import Dataset

# TODO
# 1. Generate here the trajectories in the format <frame_id> <object_id> <x> <y> instead of storing as well
#    as the sequence separators
# 2. List comprehension or vectorize relative displacements

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

class ArgoverseMotionForecastingDataset(Dataset):

    def __init__(self, trajectory_file, sequence_separators_file, obs_len=20, pred_len=30, skip=1, threshold=0.002,
                 min_objs=0, windows_frames=None, phase='train', delim='\t', frames_path="", frames_extension='png', num_agents=10):
        """
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
        - frames_path: Path to physical information 
        - frames_extension: png, npy (e.g. HD maps), jpg, etc.
        - num_agents: Number of agents to be considered in a single forward (including the ego-vehicle). If there are less than num_agents,
          dummy variables are used to predict. If there are more, agents are filtered by its distance to the ego-vehicle
          forwards but using the same physical information and frame index
        """

        # Initialize variables

        super(ArgoverseMotionForecastingDataset, self).__init__()
        self.dataset_name = "argoverse_motion_forecasting_dataset"
        self.objects_id_dict = {"Car": 0, "Cyc": 1, "Mot": 2, "Ped": 3} # TODO: Get this by argument
        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.skip, self.delim = skip, delim
        self.frames_path = frames_path

        self.repo_folder = "/home/robesafe/libraries/SoPhie/"
        self.parent_folder = "data/datasets/argoverse/motion-forecasting/train/"
        """
        # Load trajectory_file and sequence_separators
        
        with open(trajectory_file, 'rb') as obs_file:
          joined_obs_trajectories =  np.load(obs_file)
          print("Trajectories shape: ", joined_obs_trajectories.shape)

        with open(sequence_separators_file, 'rb') as seq_sep_file:
          seq_separators = np.load(seq_sep_file).reshape(-1)
          print("Separators shape: ", seq_separators.shape)

        # Get maximum distance between AV and AGENT for the last observation of the past positions

        max_distance = 0
        index_max_distance = 0

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

          if dist > max_distance:
            max_distance = dist
            index_max_distance = i

        print("Max distance: ", max_distance)
        print("Index max distance: ", index_max_distance)

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
        """
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











        for path in all_files:
            print_str = 'load %s\r' % path
            sys.stdout.write(print_str)
            sys.stdout.flush()

            _, seq_name, _ = fileparts(path)

            print(">> ", path, seq_name)
            data = read_file(path, delim)

            # Testing files only contains past, so add more windows

            if phase == 'test':
                min_frame, max_frame = 0, 999
                num_windows = int(max_frame - min_frame + 1 - skip*(self.seq_len - 1))      
                num_windows += (self.pred_len-1)*skip + 1
            else:
                frames = np.unique(data[:, 0]).tolist()
                min_frame, max_frame = frames[0], frames[-1]
                num_windows = int(max_frame - min_frame + 1 - skip*(self.seq_len - 1)) # Include all frames for past and future 

            # Loop every window

            for window_index in range(num_windows):
                start_frame = int(window_index + min_frame)
                end_frame = int(start_frame + self.seq_len*skip) # This frame is not included

                if phase == 'test':
                    if windows_frames and start_frame not in windows_frames:
                        continue
                
                frame = start_frame + self.obs_len # Frame of interest

                seq_name_int = seqname2int(seq_name)
                if frame > 999: # ¿?¿?¿? if and only if you have 1000 frames at most
                    frame -= 1
                seq_frame = np.array([seq_name_int, frame])

                # Reduce window during testing, only evaluate every N windows
                # if phase == 'test':
                #     check_pass = check_eval_windows(start_frame+self.obs_len*skip, self.obs_len*skip, self.pred_len*skip, phase=phase)
                #     if not check_pass: 
                #         continue

                # Get data in current window

                curr_seq_data = []
                for frame in range(start_frame, end_frame, skip):
                    curr_seq_data.append(data[frame == data[:,0],:])
                curr_seq_data = np.concatenate(curr_seq_data, axis=0) # Frame - ID - X - Y

                # Initialize data

                objs_in_curr_seq_list = []
                objs_in_curr_seq = np.unique(curr_seq_ata[:,1]) # Number of objects in the window
                objs_len = objs_in_curr_seq.shape[0]

                num_mini_batches = math.ceil(float(objs_len/num_agents)) # If there are less than num_agents,
                # dummy variables are used to predict. If there are more, it is distributed in total_agents % num_agents
                # forwards but using the same physical information and frame index

                for mini_batch_index in range(num_mini_batches):
                    if mini_batch == num_mini_batches - 1:
                        objs_in_curr_seq_aux = objs_in_curr_seq[num_agents*mini_batch_index:]
                        dummy = [-1 for i in range(num_agents - objs_in_curr_seq_aux.shape[0])]
                        objs_in_curr_seq_aux = np.concatenate((objs_in_curr_seq_aux, dummy))
                    else:
                        objs_in_curr_seq_aux = objs_in_curr_seq[num_agents*mini_batch_index:num_agents*(mini_batch_index+1)]
                    objs_in_curr_seq_list.append(objs_in_curr_seq_aux)

                # Create final data to feed the model based on objs_in_curr_seq_list

                for current_objs in objs_in_curr_seq_list:
                    curr_seq_rel   = np.zeros((len(current_objs), 2, self.seq_len)) # num_agents x 2 x seq_len
                    curr_seq       = np.zeros((len(current_objs), 2, self.seq_len)) # num_agents x 2 x seq_len
                    curr_loss_mask = np.zeros((len(current_objs),    self.seq_len)) # num_agents x seq_len
                    id_frame_list  = np.zeros((len(current_objs), 3, self.seq_len)) # num_agents x 3 x seq_len
                    object_class_list = np.zeros(len(current_objs))
                    object_class_list.fill(-1)
                    id_frame_list.fill(0)
                    num_objs_considered = 0
                    _non_linear_obj = []

                    # Loop through every object in this window

                    for _, obj_id in enumerate(current_objs):
                        if obj_id == -1:
                            num_objs_considered += 1
                            continue

                        object_class = getObjecClass(path)
                        object_class_id = self.objects_id_dict[object_class]
                        curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :] # Frame - ID - X - Y for one of the id of the window, same id
                        pad_front    = int(curr_obj_seq[0, 0] ) - start_frame      # First frame of window       
                        pad_end      = int(curr_obj_seq[-1, 0]) - start_frame + skip # Last frame of window

                        assert pad_end % skip == 0, 'error'

                        frame_existing = curr_obj_seq[:, 0].tolist() # Frames of windows

                        # Pad front and end data to make the trajectory complete

                        if ((pad_end - pad_front) != (self.seq_len * skip)):
                            # Pad front
                            to_be_paded_front = int(pad_front / skip)
                            pad_front_seq = np.expand_dims(curr_obj_seq[0, :], axis=0)
                            pad_front_seq = np.repeat(pad_front_seq, to_be_paded_front, axis=0)
                            frame_offset = np.zeros((to_be_paded_front, 4), dtype='float32')
                            frame_offset[:, 0] = np.array(range(-to_be_paded_front, 0))
                            pad_front_seq += frame_offset * skip
                            curr_obj_seq = np.concatenate((pad_front_seq, curr_obj_seq), axis=0)

                            # Pad end
                            to_be_paded_end = int(self.seq_len - pad_end / skip)
                            pad_end_seq = np.expand_dims(curr_obj_seq[-1, :], axis=0)
                            pad_end_seq  = np.repeat(pad_end_seq, to_be_paded_end, axis=0)
                            frame_offset = np.zeros((to_be_paded_end, 4)), dtype='float32')
                            frame_offset[:, 0] = np.array(range(1, to_be_paded_end+1))
                            pad_end_seq += frame_offset * skip # Shift first columns for frame
                            curr_obj_seq = np.concatenate((curr_obj_seq, pad_end_seq), axis=0)

                            # Set pad front and end to correct values
                            pad_front = 0
                            pad_end = self.seq_len * skip

                        # Special cases

                        # A. The object reappears at a bad frame (missing intermediate frame)

                        if curr_obj_seq.shape[0] != (pad_end - pad_front) / skip:
                            frame_all = list(range(int(curr_obj_seq[0,0]), int(curr_obj_seq[-1,0]) + skip, skip))
                            frame_missing, _ = remove_list_from_list(frame_all, curr_obj_seq[:, 0].tolist())

                            # Pad all missing frames with zeros

                            pad_seq = np.expand_dims(curr_obj_seq[-1,:], axis=0)
                            pad_seq = np.repeat(pad_seq, len(frame_missing), axis=0)
                            pad_seq.fill(0)
                            pad_seq[:,0] = np.array(frame_missing)
                            pad_seq[:,1] = obj_id # Fill ID
                            curr_obj_seq = np.concatenate((curr_obj_seq, pad_seq), axis=0)
                            curr_obj_seq = curr_obj_seq[np.argsort(curr_obj_seq[:,0])]

                        assert pad_front == 0, 'error'
                        assert pad_end == self.seq_len * skip, 'error'

                        # B. Make sure the seq_len frames are continuous, no jumping frames

                        start_frame_now = int(curr_obj_seq[0,0])
                        if curr_obj_seq[-1,0] != start_frame_now + (self.seq_len-1)*skip:
                            num_objs_considered += 1
                            continue

                        # C. Make sure the past data has at least one frame

                        past_frame_list = [*range(start_frame_now, start_frame_now + self.obs_len * skip, skip)]
                        common = find_unique_common_from_lists(past_frame_list, frame_existing, only_com=True)
                        if len(common) == 0:
                            num_objs_considered += 1
                            continue

                        # D. Make sure that future GT data has at least one frame

                        if phase != 'test':
                            gt_frame_list = [*range(start_frame_now + self.obs_len*skip, start_frame_now + self.seq_len*skip, skip)]
                            common = find_unique_common_from_lists(gt_frame_list, frame_existing, only_com=True)
                            if len(common) == 0:
                                num_objs_considered += 1
                                continue

                        # Only keep the state

                        cache_tmp = np.transpose(curr_obj_seq[:,:2]) # 2 x seq_len | [0,:] list of frames | [1,:] id
                        curr_obj_seq = np.transpose(curr_obj_seq[:,2:]) # 2 x seq_len | [0,:] x | [1,:] y

                        # Make coordinates relative

                        rel_curr_obj_seq = np.zeros(curr_obj_seq.shape)
                        rel_curr_obj_seq[:,1:] = curr_obj_seq[:,1:] - curr_obj_seq[:,:-1]
                        _idx = num_objs_considered
                        curr_seq[_idx, :, :] = curr_obj_seq
                        curr_seq_rel[_idx, :, :] = rel_curr_obj_seq

                        # Record seqname, frame and ID information 

                        id_frame_list[_idx, :2, :] = cache_tmp
                        id_frame_list[_idx,  2, :] = seq_name_int # img_id - obj_id - seqname2int

                        # Linear vs Non-Linear Trajectory, only fit for the future part, not past part

                        if phase != 'test':
                            _non_linear_obj.append(poly_fit(curr_obj_seq, pred_len, threshold))

                        # Add mask onto padded dummy data

                        frame_exist_index = np.array([frame_tmp - start_frame_now for frame_tmp in frame_existing])
                        frame_exist_index = (frame_exist_index / skip).astype('uint8')
                        curr_loss_mask[_idx, frame_exist_index] = 1

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
                        frames_list.append(seq_frame)
                        object_class_id_list.append(object_class_list)
                        object_id_list.append(id_frame_list[:num_objs_considered][:,1,0])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # Objects x 2 x seq_len
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_obj = np.asarray(non_linear_obj)
        seq_id_list = np.concatenate(seq_id_list, axis=0)
        frames_list = np.asarray(frames_list)
        object_class_id_list = np.asarray(object_class_id_list)
        object_id_list = np.asarray(object_id_list)

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
        self.frames_list = torch.from_numpy(frames_list).type(torch.float)
        self.object_class_id_list = torch.from_numpy(object_class_id_list).type(torch.float)
        self.object_id_list = torch.from_numpy(object_id_list).type(torch.float)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
                self.obs_traj[start:end, :], self.pred_traj_gt[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel_gt[start:end, :],
                self.non_linear_obj[start:end], self.loss_mask[start:end, :],
                self.seq_id_list[start:end, :], self.frames_path, self.frames_extension, self.frames_list[index, :],
                self.object_class_id_list[index], self.object_id_list[index]
              ]   

        return out

