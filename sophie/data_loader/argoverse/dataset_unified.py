#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 16 15:45:01 2021
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

import psutil
import copy
import cv2
import glob, glob2
import json
import csv
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
from numba import jit
import pdb

from pathlib import Path
from torch.utils.data import Dataset

lib_path = os.path.join("/home", "robesafe", "libraries")
if os.path.exists(lib_path):
    sys.path.append("/home/robesafe/libraries/SoPhie")
from argoverse.map_representation.map_api import ArgoverseMap
import sophie.data_loader.argoverse.map_utils as map_utils

# Global variables

frames_path = None
avm = ArgoverseMap()
dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]

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
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                full_list += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            full_list += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            full_list += newlist

    full_list = [os.path.normpath(path_tmp) for path_tmp in full_list]
    num_elem = len(full_list)

    return full_list, num_elem

def get_sequences_as_array(root_folder="data/datasets/argoverse/motion-forecasting/",split="train",
                           obs_len=20,pred_len=30,distance_threshold=35,num_agents_per_obs=10,split_percentage=1.0):
    """
	Input: Directory containing the main files (TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME)
    """

    folder = root_folder + split + "/data/"
    ids_file = root_folder + split + "/ids_file.json"
    parent_folder = '/'.join(os.path.normpath(folder).split('/')[:-1])

    start = time.time()

    files, num_files = load_list_from_folder(folder)
    file_id_list = []
    for file_name in files:
        file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
        file_id_list.append(file_id)
    file_id_list.sort()
    print("Num files: ", num_files)

    end = time.time()
    # print(f"Time consumed listing the files: {end-start}\n") # 1s

    # Flags

    study_maximum_distance = False 

    if not Path(ids_file).is_file():
        print("Calculating JSON file with IDs equivalence ...")
        start = time.time()

        track_id_coded = dict()

        for i,file_id in enumerate(file_id_list):
            track_file = folder + str(file_id) + ".csv"
            # print("--------------------")
            print(f"File {i+1}/{num_files}")
            
            # Get AV and AGENT ids in Argoverse format

            data = csv.DictReader(open(track_file)) # Faster than using pandas.read_csv
            agents = dict()
            cont = 2
            for row in data:
                if row['TRACK_ID'] not in agents.values():
                    if row['OBJECT_TYPE'] == 'AV':
                        agents[0] = row['TRACK_ID']         
                    elif row['OBJECT_TYPE'] == 'AGENT':
                        agents[1] = row['TRACK_ID']
                    elif row['OBJECT_TYPE'] == 'OTHERS':
                        agents[cont] = row['TRACK_ID']
                        cont += 1

            track_id_coded[track_file] = agents

        # Store JSON

        with open(ids_file, "w") as outfile:
            json.dump(track_id_coded, outfile)

        end = time.time()
        print(f"Time consumed storing the JSON file with the IDs equivalence: {end-start}\n")
    else:
        print(f"JSON file with IDs equivalence already generated. Placed in: {ids_file}")

    # Load JSON file with the encoding of IDs

    with open(ids_file) as input_file:
        encoding_dict = json.load(input_file)

    # Perform a study of the maximum distance of the agent in the last observation 
    # to filter the obstacles

    if study_maximum_distance:
        print("Studying the distances ego_vehicle-agent in the last observation frame") # TODO: AGENT-others?
        start = time.time()

        distances = []
        max_distance = 0
        index_max_distance = 0

        for i,file_id in enumerate(file_id_list):
            print(f"File {i+1}/{len(file_id_list)}")
            track_file = folder + str(file_id) + ".csv"
            
            data = csv.DictReader(open(track_file))

            cont = -1
            start_last_observation_index, end_last_observation_index = 0,0

            last_observation = np.zeros([0,6])
            for row in data:
                if row["OBJECT_TYPE"] == "AV":
                    cont += 1

                if cont == obs_len:
                    break
                elif cont >= obs_len - 1:
                    row = list(row.values())
                    last_observation = np.vstack((last_observation,row))

            agent_index = np.where(last_observation[:,2] == "AGENT")
            agent_x, agent_y = float(last_observation[agent_index, 3]), float(last_observation[agent_index, 4])

            current_distances = np.array([math.sqrt(pow(float(x)-agent_x,2)+pow(float(y)-agent_y,2)) for _,_,_,x,y,_ in last_observation]).reshape(-1,1)
            distances.append(current_distances)

        distances = np.concatenate(distances,axis=0)

        print("Mean distance: ", distances.mean()) # > 30 m 
        print("Min distance: ", distances.min()) # < 5 m
        print("Max distance: ", distances.max()) # > 150 m
        print("Index max distance: ", index_max_distance)

        further_agents = len(np.where(distances > distances.mean())[0])
        print("\nNum agents further than mean: ", further_agents)
        print("\nNum agents further than threshold: ", distance_threshold)

        end = time.time()

        print(f"Time consumed studying the maximum distance: {end-start}\n")

        n, bins, patches = plt.hist(distances, bins=20)
        plt.show()

        assert 1 == 0

    start = time.time()

    distance_threshold = distance_threshold # If the obstacle is further than this distance in the last
                                            # observation, discard for the whole sequence

    num_sequences_percentage = int(num_files * split_percentage)
    filtered_sequences = []
    origin = [] # Agent's position in the last observation
    city_ids = []

    for i,file_id in enumerate(file_id_list):
        if i == num_sequences_percentage:
            break

        print(f"File {i+1}/{num_sequences_percentage}")
        track_file = folder + str(file_id) + ".csv" 
        dict1 = encoding_dict[track_file] # Standard IDs (Keys) to Argoverse values (Values)
        dict2 = dict(zip(dict1.values(), dict1.keys())) # Argoverse values (Keys) to standard IDs (Values)

        data = csv.DictReader(open(track_file))

        aux = []
        for row in data:
            values = list(row.values())
            aux.append(values)
        data = np.array(aux)

        timestamps = data[:,0].astype('float').reshape(-1,1)

        track_ids = data[:,1].reshape(-1,1)
        coded_track_ids = np.vectorize(dict2.get)(track_ids).astype(np.int64)

        x_pos = data[:,3].astype('float').reshape(-1,1)
        y_pos = data[:,4].astype('float').reshape(-1,1)

        object_type = data[:,2].reshape(-1,1)
        object_class = [0 if obj=="AV" else 1 if obj=="AGENT" else 2 for obj in object_type]
        object_class = np.array(object_class).reshape(-1,1)

        sequence = (timestamps,coded_track_ids,x_pos,y_pos,object_class) # Timestamp - ID - X - Y - Class
        sequence = np.concatenate(sequence,axis=1)

        ## Get city

        city = data[0,5].astype('str').reshape(-1,1)

        if city == "PIT":
            city_ids.append(0)
        else:
            city_ids.append(1)

        # Distance filter

        print(f"Filtering objects further than {distance_threshold} around the agent in the last observation")

        # TODO: Consider this distance filter only around the AGENT? Study this distance?

        av_indices = np.where(sequence[:,1] == 0)[0]
        av_last_observation = av_indices[obs_len - 1]
        av_x, av_y = sequence[av_last_observation,2:4]
        
        if split == "test":
            last_observation = sequence[av_indices[obs_len - 1]:,:]
        else:
            last_observation = sequence[av_indices[obs_len - 1]:av_indices[obs_len],:]

        agent_index = np.where(last_observation[:,1] == 1)
        origin_agent_x, origin_agent_y = last_observation[agent_index, 2], last_observation[agent_index, 3]

        distances = np.array([math.sqrt(pow(x-origin_agent_x,2)+pow(y-origin_agent_y,2)) for _,_,x,y,_ in last_observation])
        distances[0] = -1 # We keep the AV data regardless it is further than distance_threshold. It is useful to know
                          # when each timestamp starts
        distance_indexes = np.where(distances <= distance_threshold)[0]
        relevant_obstacles_ids = last_observation[distance_indexes,1]
        relevant_obstacles_indexes = np.where(np.in1d(sequence[:,1],relevant_obstacles_ids))
        sequence = np.take(sequence, relevant_obstacles_indexes, axis=0).reshape(-1,sequence.shape[1])

        av_indices = np.where(sequence[:,1] == 0)[0]
        if split == "test":
            last_observation = sequence[av_indices[obs_len - 1]:,:]
        else:
            last_observation = sequence[av_indices[obs_len - 1]:av_indices[obs_len],:]

        av_last_observation = av_indices[obs_len - 1]
        av_x, av_y = sequence[av_last_observation,2:4]

        distances = np.array([math.sqrt(pow(x-av_x,2)+pow(y-av_y,2)) for _,_,x,y,_ in last_observation])
        distance_indexes = np.where(distances <= distance_threshold)[0]
        relevant_obstacles_ids = last_observation[distance_indexes,1]
        


        # Dummy filter (take the num_agents_per_obs-2 closest to the AGENT if there are more
        # num_agents_per_obs-2 in the corresponding timestamp (we assume the AGENT and the AV
        # are already included))

        # TODO: What is better, taking into account num_agents_per_obs vehicles if there are far away
        # from the agent, or consider a distance threshold and include dummy data?
        # Which option influences more in the behaviour of the agent?

        dummy_filtered_sub_sequences = []
        av_indices = np.where(sequence[:,1] == 0)[0]

        for j in range(len(av_indices)):
            start_window_index = av_indices[j]
            if j < len(av_indices) - 1: 
                end_window_index = av_indices[j+1]
            else:
                end_window_index = sequence.shape[0]

            agents_in_obs = end_window_index - start_window_index
            sub_sequence = sequence[start_window_index:end_window_index,:]

            if agents_in_obs < num_agents_per_obs: # We introduce dummy data
                timestamp = sequence[start_window_index,0]
                dummy_agents = num_agents_per_obs - agents_in_obs
                dummy_array = np.array([timestamp,-1,-1.0,-1.0,-1]) # Timestamp - ID - X - Y - Class
                dummy_array = np.tile(dummy_array,dummy_agents).reshape(-1,5)
                dummy_sub_sequence = np.concatenate([sub_sequence,dummy_array])
            elif agents_in_obs == num_agents_per_obs:
                dummy_sub_sequence = sub_sequence
            else: # Get the num_agents_per_obs closest vehicles to the agent
                agents_dist = np.zeros((sub_sequence.shape[0]))
                agent_index = np.where(sub_sequence[:,1] == 1)
                agent_x, agent_y = sub_sequence[agent_index, 2], sub_sequence[agent_index, 3]

                for k in range(sub_sequence.shape[0]):
                    # The AV (ID=0) starts the observations of each timestamp
                    if sub_sequence[k,1] != 0 and sub_sequence[k,1] != 1: # TODO: Discard ego-vehicle (AV) position if it is not
                        # one of the num_agents_per_obs-1 closest obstacles?

                        obj_x = sub_sequence[k,2]
                        obj_y = sub_sequence[k,3]

                        dist = math.sqrt(pow(obj_x-agent_x,2)+pow(obj_y-agent_y,2))
                        agents_dist[k] = dist

                sorted_indeces = np.argsort(agents_dist)
                to_delete_indeces = sorted_indeces[num_agents_per_obs:] # Only keep the closest num_agents_per_obs agents

                dummy_sub_sequence = np.delete(sub_sequence,to_delete_indeces,axis=0)
            dummy_filtered_sub_sequences.append(dummy_sub_sequence)
        sequence = np.concatenate(dummy_filtered_sub_sequences)
        # print("Dummy filtered sequences: ", sequence, sequence.shape)

        # av_indexes = np.where(sequence[:,1] == 1)
        # print("PRE GET SEQUENCES av indexes: ", av_indexes)
        # x1, y1 = sequence[av_indexes[0][0],2], sequence[av_indexes[0][0],3]
        # x2, y2 = sequence[av_indexes[0][-1],2], sequence[av_indexes[0][-1],3]
        # print("x1, y1, x2, y2: ", x1, y1, x2, y2)
        # dist = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
        # print("Dist: ", dist)

        # Get relative displacements

        num_observations = obs_len + pred_len

        assert num_observations == sequence.shape[0] / num_agents_per_obs

        # origin_x = sequence[(obs_len-1)*num_agents_per_obs,2] # Ego-vehcile's position in last observation
        # origin_y = sequence[(obs_len-1)*num_agents_per_obs,3]
        origin_aux = np.array([origin_agent_x, origin_agent_y]).reshape(1,2)
        origin.append(origin_aux)

        other_obstacles_indices = np.where(sequence[:,1] != -1)[0]
        sequence[other_obstacles_indices,2:4] -= origin_aux

        av_indexes = np.where(sequence[:,1] == 1)
        # print("\n\nGET SEQUENCES av indexes: ", av_indexes)
        x1, y1 = sequence[av_indexes[0][0],2], sequence[av_indexes[0][0],3]
        x2, y2 = sequence[av_indexes[0][-1],2], sequence[av_indexes[0][-1],3]
        dist = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))

        av_indeces = np.where(sequence[:,1] == 0)[0]
        agent_indeces = np.where(sequence[:,1] == 1)[0]

        # Append filtered sequences

        filtered_sequences.append(sequence)

    sequences = np.dstack(filtered_sequences) # Get sequence_info = seq_len * num_agents_per_obs (e.g. obs_len+pred_len = 50 x num_agents_per_obs = 10) x info (= 5 -> Timestamp | ID | X | Y | Class) x Num_sequences (Files of the folder)
    origin = np.dstack(origin) # 1 x 2 (X | Y) x Num_sequences
    city_ids = np.array(city_ids)

    # print("Num sequences (files): ", sequences.shape)
    # print("Origin: ", origin.shape)
    # print("City ids: ", city_ids.shape)

    relative_sequences_file = "/home/robesafe/shared_home/test_map_argoverse/new_relative_sequences_3.npy"
    with open(relative_sequences_file, 'wb') as file:
        np.save(relative_sequences_file, sequences)

    origin_file = "/home/robesafe/shared_home/test_map_argoverse/new_origin_3.npy"
    with open(origin_file, 'wb') as file:
        np.save(origin_file, origin)

    # assert 1 == 0

    end = time.time()
    print(f"Time consumed filtering the sequences: {end-start}\n")
    return sequences, origin, city_ids, file_id_list

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

def load_images(num_seq, obs_seq_data, city_id, ego_origin, dist_rasterized_map, num_agents_per_obs, objs_id_list,debug_images=False):
    """
    Get the corresponding rasterized map
    """

    print("LOAD IMAGES")

    batch_size = int(obs_seq_data.shape[1]/num_agents_per_obs)
    frames_list = []

    # rasterized_start = time.time()

    for i in range(batch_size):
        curr_num_seq = int(num_seq[i].cpu().data.numpy())
        obj_id_list = objs_id_list[i].cpu().data.numpy()

        if i < batch_size - 1:
            curr_obs_seq_data = obs_seq_data[:,num_agents_per_obs*i:num_agents_per_obs*(i+1),:]
        else:
            curr_obs_seq_data = obs_seq_data[:,num_agents_per_obs*i:,:]

        curr_city = city_id[i]
        if curr_city == 0:
            city_name = "PIT"
        else:
            city_name = "MIA"

        curr_ego_origin = ego_origin[i].reshape(1,-1)
        start = time.time()

        curr_obs_seq_data = curr_obs_seq_data.reshape(-1,2) # Past_Observations x Num_Agents x 2 -> (Past_Observations * Num_agents) x 2 
                                                            # (required by map_utils)

        print("Curr obs seq data: ", curr_obs_seq_data, curr_obs_seq_data.shape)
        print("curr_ego_origin: ", curr_ego_origin, curr_ego_origin.shape)

        fig = map_utils.map_generator(
            curr_obs_seq_data, curr_ego_origin, dist_rasterized_map, avm, city_name,
            (obj_id_list, num_agents_per_obs), show=False, smoothen=True
        )
        end = time.time()
        # print(f"Time consumed by map generator: {end-start}")
        start = time.time()
        img = map_utils.renderize_image(fig)

        if debug_images:
            # print("img: ", type(img), img.shape)
            print("frames path: ", frames_path)
            print("curr seq: ", str(curr_num_seq))
            filename = frames_path + "seq_" + str(curr_num_seq) + ".png"
            print("path: ", filename)
            img = img * 255.0
            cv2.imwrite(filename,img)

        plt.close("all")
        end = time.time()
        frames_list.append(img)
        # print(f"Time consumed by map render: {end-start}")

    # rasterized_end = time.time()
    # print(f"Time consumed by rasterized image: {rasterized_end-rasterized_start}")

    frames_arr = np.array(frames_list)
    return frames_arr

def seq_collate(data):
    """
    This functions takes as input the dataset output (see __getitem__ function below) and transforms it to
    a particular format to feed the Pytorch standard dataloader

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    non_linear_obj, loss_mask, seq_id_list, object_class_id_list, 
    object_id_list, city_id, ego_origin, num_seq_list)

                                    |
                                    v

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    non_linear_obj, loss_mask, seq_start_end, frames, "object_cls", 
    "obj_id", ego_vehicle_origin, num_seq_list) = batch
    """

    start = time.time()

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_obj, loss_mask, seq_id_list, object_class_id_list, 
     object_id_list, city_id, ego_vehicle_origin, num_seq_list) = zip(*data)

    batch_size = len(ego_vehicle_origin) # tuple of tensors

    _len = [len(seq) for seq in obs_traj]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size

    obs_traj = torch.cat(obs_traj, dim=0).permute(2, 0, 1) # Past Observations x Num_agents · batch_size x 2
    pred_traj_gt = torch.cat(pred_traj_gt, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=0).permute(2, 0, 1)
    pred_traj_gt_rel = torch.cat(pred_traj_gt_rel, dim=0).permute(2, 0, 1)
    non_linear_obj = torch.cat(non_linear_obj)
    loss_mask = torch.cat(loss_mask, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    id_frame = torch.cat(seq_id_list, dim=0).permute(2, 0, 1) # seq_len - objs_in_curr_seq - 3

    start = time.time()
    num_agents_per_obs = int(obs_traj.shape[1] / batch_size)
    print("PRE LOAD")
    frames = load_images(num_seq_list, obs_traj_rel, city_id, ego_vehicle_origin,    # Return batch_size x 600 x 600 x 3
                         dist_rasterized_map, num_agents_per_obs, object_id_list, debug_images=True)
    
    end = time.time()
    # print(f"Time consumed by load_images function: {end-start}\n")

    frames = torch.from_numpy(frames).type(torch.float32)
    frames = frames.permute(0, 3, 1, 2)

    object_cls = torch.stack(object_class_id_list)
    obj_id = torch.stack(object_id_list)
    ego_vehicle_origin = torch.stack(ego_vehicle_origin)
    num_seq_list = torch.stack(num_seq_list)

    out = [obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
           loss_mask, seq_start_end, frames, object_cls, obj_id, ego_vehicle_origin, num_seq_list] 

    end = time.time()
    # print(f"Time consumed by seq_collate function: {end-start}\n")

    return tuple(out)

class ArgoverseMotionForecastingDataset(Dataset):
    def __init__(self, dataset_name, root_folder, obs_len=20, pred_len=30, skip=1, threshold=0.002, distance_threshold=30,
                 min_objs=0, windows_frames=None, split='train', num_agents_per_obs=10, split_percentage=0.1):
        """
        - root_folder: Directory containing the main files 
        - obs_len: Number of observed frames in prior (input) trajectories
        - pred_len: Number of predicted frames in posterior (output) trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using a linear predictor
        - distance_threshold: If an obstacle is further away than distance threshold in the obs_len-th frame, it is discarded in
                              the entire sequence
        - min_objs: Minimum number of objects that should be in a sequence
        - windows_frames: Specific frames to analize the past and predict the future from that point
        - split: train/val (GT is provided) or test (GT not provided)
        - num_agents: Number of agents to be considered in a single forward (including the ego-vehicle). If there are less than num_agents,
            dummy variables are used to predict. If there are more, agents are filtered by its distance to the ego-vehicle
            forwards but using the same physical information and frame index
        - split_percentage: Percentage of the corresponding split [0.0 to 1.0] to be used (either training or testing)
        """

        # Initialize variables

        super(ArgoverseMotionForecastingDataset, self).__init__()
        self.root_folder = root_folder
        self.dataset_name = dataset_name
        self.objects_id_dict = {"DUMMY": -1, "AV": 0, "AGENT": 1, "OTHERS": 2} # TODO: Get this by argument
        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.skip = skip
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.min_objs = min_objs
        self.windows_frames = windows_frames
        self.split = split

        global frames_path
        if os.path.exists(lib_path):
            frames_path = "/home/robesafe/libraries/SoPhie/data/datasets/argoverse/motion-forecasting/" + split + "/data_images/" 
        else:
            frames_path = "/home/robesafe/tesis/SoPhie/data/datasets/argoverse/motion-forecasting/" + split + "/data_images/" 
        if not os.path.exists(frames_path):
            print("Create frames folder: ", frames_path)
            os.mkdir(frames_path)

        self.num_agents_per_obs = num_agents_per_obs
        self.split_percentage = split_percentage

        if self.split == "test":
            self.pred_len = 0           # Must be 0 for the split test since we do not have the predictions of the agents
            self.seq_len = self.obs_len # Only the observations (0 to obs_len-1 past observations)

        num_objs_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_obj = []
        seq_id_list = []
        num_seq_list = []
        frames_list = []
        object_class_id_list = []
        object_id_list = []

        # Get the normalized sequences ang ego_vehicle_origin for each file

        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"1. Load files and filter sequences of {self.split} split")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

        sequences, self.ego_vehicle_origin, self.city_ids, file_id_list = get_sequences_as_array(root_folder=self.root_folder,
                                                                                                 split=self.split,
                                                                                                 obs_len=self.obs_len,
                                                                                                 pred_len=self.pred_len,
                                                                                                 distance_threshold=self.distance_threshold,
                                                                                                 num_agents_per_obs=self.num_agents_per_obs,
                                                                                                 split_percentage=self.split_percentage)

        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("2. Get the corresponding data to feed the GAN-LSTM network for Motion Prediction")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        start = time.time()

        for seq_index in range(sequences.shape[2]):
            print(f"File {seq_index+1}/{sequences.shape[2]}")

            num_seq_list.append(seq_index)

            if seq_index < sequences.shape[2] - 1:
                curr_seq_data = sequences[:,:,seq_index:seq_index+1] # Frame - ID - X - Y - Class
            else:
                curr_seq_data = sequences[:,:,seq_index:] # Frame - ID - X - Y - Class

            rows, columns, depth = curr_seq_data.shape # (seq_len * num_agents_per_obs) x 5 x 1
            curr_seq_data = curr_seq_data.reshape(rows,columns) # From tensor (3D) to matrix (2D)

            av_indexes = np.where(curr_seq_data[:,1] == 0)

            curr_seq_timestamps = curr_seq_data[:,0]
            curr_seq_timestamps = curr_seq_timestamps[::self.num_agents_per_obs] # Take the timestamp per obs (each obs has self.num_agents_per_obs)
                                                                                 # salto de self.num_agents_per_obs

            # From relative to global coordinates
            ego_origin_seq = self.ego_vehicle_origin[0,:,seq_index].reshape(1,2)
            curr_seq_data[:,2:4] += ego_origin_seq
            ego_origin_seq = ego_origin_seq.reshape(2,1)

            # curr_seq_data dimensions: Rows (= Num_agents_per_obs x Num_obs) x Columns (5)

            # Initialize data

            curr_seq_rel   = np.zeros((self.num_agents_per_obs, 2, self.seq_len)) # num_agents x 2 x seq_len
            curr_seq       = np.zeros((self.num_agents_per_obs, 2, self.seq_len)) # num_agents x 2 x seq_len
            curr_loss_mask = np.zeros((self.num_agents_per_obs,    self.seq_len)) # num_agents x seq_len
            id_frame_list  = np.zeros((self.num_agents_per_obs, 3, self.seq_len)) # num_agents x 3 x seq_len
            object_class_list = np.zeros(self.num_agents_per_obs)
            object_class_list.fill(-1) # By default, -1 (dummy agents)
            id_frame_list.fill(-1)
            num_objs_considered = 0
            _non_linear_obj = []

            objs_in_curr_seq = np.unique(curr_seq_data[:,1]) # Number of objects in the window

            # Add dummy agents if unique agents < num_agents

            if objs_in_curr_seq.shape[0] < self.num_agents_per_obs:
                diff = self.num_agents_per_obs - objs_in_curr_seq.shape[0]
                dummy_ids = -1 * np.ones((diff))
                objs_in_curr_seq = np.hstack([dummy_ids,objs_in_curr_seq])

            # Loop through every object in this window

            num_dummies = 0

            for obj_index, obj_id in enumerate(objs_in_curr_seq):
                if obj_id == -1:
                    _idx = num_objs_considered
                    x_dummy = np.random.rand(1,self.seq_len)
                    y_dummy = np.random.rand(1,self.seq_len)
                    curr_obj_seq = np.vstack([x_dummy,y_dummy]) # 2 x 50
                    rel_curr_obj_seq = curr_obj_seq - ego_origin_seq

                    curr_seq[_idx, :, :] = curr_obj_seq
                    curr_seq_rel[_idx, :, :] = rel_curr_obj_seq
                    num_objs_considered += 1
                    continue
                elif obj_index > self.num_agents_per_obs - 1:
                    continue

                object_indexes = np.where(curr_seq_data[:,1]==obj_id)[0]
                num_obs_obj_id = len(object_indexes)

                object_id_index = object_indexes[0] # First index in which this object appears
                object_class_id = curr_seq_data[object_id_index,4]

                curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :] # Frame - ID - X - Y - Class  for one of the id of the window, same id
                                                                               # 50 x 5
                # Only keep the state

                cache_tmp = np.transpose(curr_obj_seq[:,:2]) # 2 x seq_len -> First row = [0,:] list of timestamps | Second row = [1,:] id
                curr_obj_seq = np.transpose(curr_obj_seq[:,2:-1]) # 2 x seq_len -> First row = [0,:] x | Second row = [1,:] y
                                                                  # Global coordinates
                
                if num_obs_obj_id < self.seq_len:
                    object_indexes = np.array([])
                    num_dummies += 1
                    id_dummy = -1 * np.ones((1,self.seq_len))
                    cache_tmp = np.vstack([curr_seq_timestamps,id_dummy])

                    x_dummy = np.random.rand(1,self.seq_len)
                    y_dummy = np.random.rand(1,self.seq_len)
                    curr_obj_seq = np.vstack([x_dummy,y_dummy])

                    object_class_id = -1
                                                                  
                # Make coordinates relative

                rel_curr_obj_seq = curr_obj_seq - ego_origin_seq
                _idx = num_objs_considered
                curr_seq[_idx, :, :] = curr_obj_seq
                curr_seq_rel[_idx, :, :] = rel_curr_obj_seq

                # Record seqname, frame and ID information 

                id_frame_list[_idx, :2, :] = cache_tmp
                id_frame_list[_idx,  2, :] = file_id_list[seq_index] 

                # Linear vs Non-Linear Trajectory, only fit for the future part, not past part

                if self.split != 'test':
                    _non_linear_obj.append(poly_fit(curr_obj_seq, pred_len, threshold))

                # Add mask onto padded dummy data

                object_indexes_obs = (object_indexes / self.num_agents_per_obs).astype(np.int64)
                curr_loss_mask[_idx, object_indexes_obs] = 1

                # Object ID

                object_class_list[num_objs_considered] = object_class_id
                num_objs_considered += 1

            # assert 1 == 0
            if num_objs_considered > min_objs:
                if len(_non_linear_obj) != num_objs_considered:
                    dummy = [-1 for i in range(num_objs_considered - len(_non_linear_obj))]
                    _non_linear_obj = _non_linear_obj + dummy
                non_linear_obj += _non_linear_obj
                num_objs_in_seq.append(num_objs_considered)
                loss_mask_list.append(curr_loss_mask[:num_objs_considered])
                seq_list.append(curr_seq[:num_objs_considered]) # (x,y)
                seq_list_rel.append(curr_seq_rel[:num_objs_considered]) # (x_rel, y_rel)
                seq_id_list.append(id_frame_list[:num_objs_considered]) # (timestamp, id, file_id)
                object_class_id_list.append(object_class_list) # obj_class (-1 0 1 2 2 2 2 ...)
                object_id_list.append(id_frame_list[:,1,0])

        end = time.time()

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # Objects x 2 x seq_len
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_obj = np.asarray(non_linear_obj)
        seq_id_list = np.concatenate(seq_id_list, axis=0)
        object_class_id_list = np.asarray(object_class_id_list)
        object_id_list = np.asarray(object_id_list)
        num_seq_list = np.concatenate([num_seq_list])

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_gt = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_gt_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_obj = torch.from_numpy(non_linear_obj).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_objs_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.seq_id_list = torch.from_numpy(seq_id_list).type(torch.float)
        self.object_class_id_list = torch.from_numpy(object_class_id_list).type(torch.float)
        self.object_id_list = torch.from_numpy(object_id_list).type(torch.float)
        self.ego_vehicle_origin = torch.from_numpy(self.ego_vehicle_origin).type(torch.float)
        self.num_seq_list = torch.from_numpy(num_seq_list).type(torch.int)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
                self.obs_traj[start:end, :, :], self.pred_traj_gt[start:end, :, :],
                self.obs_traj_rel[start:end, :, :], self.pred_traj_gt_rel[start:end, :, :],
                self.non_linear_obj[start:end], self.loss_mask[start:end, :],
                self.seq_id_list[start:end, :, :], self.object_class_id_list[index], 
                self.object_id_list[index], self.city_ids[index], self.ego_vehicle_origin[0,:,index].reshape(1,2),
                self.num_seq_list[index]
              ] 

        return out
 
