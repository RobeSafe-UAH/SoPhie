#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 16 15:45:01 2021
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

import os
import sys
import glob, glob2
import copy
import time
import csv
import numpy as np
import json
import math
import pandas as pd

from pathlib import Path

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
                           obs_len=20,pred_len=30,distance_threshold=35,num_agents_per_obs=10):
    """
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
    print(f"Time consumed listing the files: {end-start}\n")

    # Flags

    study_maximum_distance = False

    if not Path(ids_file).is_file():
        start = time.time()

        track_id_coded = dict()

        for i,file_id in enumerate(file_id_list):
            track_file = folder + str(file_id) + ".csv"
            
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
        print(f"JSON file already generated. Placed in: {ids_file}")

        # Load JSON file with the encoding of IDs

        with open(ids_file) as input_file:
            encoding_dict = json.load(input_file)

        # Perform a study of the maximum distance of the agent in the last observation 
        # to filter the obstacles

        if study_maximum_distance:
            start = time.time()

            distances = np.zeros((num_files,1))
            max_distance = 0
            index_max_distance = 0

            for i,file_id in enumerate(file_id_list):
                track_file = folder + str(file_id) + ".csv"
                
                data = csv.DictReader(open(track_file))

                cont = -1
                x_av, y_av = 0, 0
                
                for row in data:
                    if (cont == obs_len - 1) and row["OBJECT_TYPE"] == "AGENT":             
                        x_agent, y_agent = float(row["X"]), float(row["Y"])
                        dist = math.sqrt(pow(x_av-x_agent,2)+pow(y_av-y_agent,2))
                        distances[i] = dist 
                        if dist > max_distance:
                            max_distance = dist
                            index_max_distance = i

                            break
                    elif row["OBJECT_TYPE"] == "AV":  
                        x_av, y_av = float(row["X"]), float(row["Y"])
                        cont += 1
                    
            print("Mean: ", distances.mean()) # > 30 m
            print("Min: ", distances.min()) # < 5 m
            print("Max: ", distances.max()) # > 150 m
            print("Index max distance: ", index_max_distance)

            end = time.time()

            print(f"Time consumed studying the maximum distance: {end-start}\n")

        start = time.time()

        distance_threshold = 35 # If the obstacle is further than this distance in the last
                                # observation, discard for the whole sequence
        sequence_list = []
        for i,file_id in enumerate(file_id_list):
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

            # print("sequence: ", sequence, sequence.shape)

            # Distance filter

            av_indices = np.where(sequence[:,1] == 0)[0]
            av_last_observation = av_indices[obs_len - 1]
            av_x, av_y = sequence[av_last_observation,2:4]

            last_observation = sequence[av_indices[obs_len - 1]:av_indices[obs_len],:]

            distances = np.array([math.sqrt(pow(x-av_x,2)+pow(y-av_y,2)) for _,_,x,y,_ in last_observation])
            distance_indexes = np.where(distances <= distance_threshold)[0]
            relevant_obstacles_ids = last_observation[distance_indexes,1]
            relevant_obstacles_indexes = np.where(np.in1d(sequence[:,1],relevant_obstacles_ids))
            
            sequence = np.take(sequence, relevant_obstacles_indexes, axis=0).reshape(-1,sequence.shape[1])
            
            print("distance filtered sequence: ", sequence, sequence.shape)

            # Dummy filter

            av_indices = np.where(sequence[:,1] == 0)[0]

            for i in range(len(av_indices)):
                start_window_index = av_indices[i]
                if i < len(av_indices) - 1: 
                    end_window_index = av_indices[i+1]
                else:
                    end_window_index = sequence.shape[0] - 1

                agents_in_obs = end_window_index - start_window_index
            

                # continueee


                if agents_in_obs < num_agents_per_obs - 1: # We introduce dummy data
                    timestamp = sub_sequence[0,0]
                    dummy_agents = num_agents_per_obs - agents_in_obs - 1
                    dummy_array = np.array([timestamp,-1,-1.0,-1.0,-1]) # timestamp, object_id, x, y, object_class
                    dummy_array = np.tile(dummy_array,dummy_agents).reshape(-1,5)
                    dummy_sub_sequence = np.concatenate([sub_sequence,dummy_array])


            # REQUIRED TO CONCATENATE. IF THE ARRAYS DO NOT HAVE THE SAME DIMENSION -> VisibleDeprecationWarning: Creating an 
            # ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray

            # Relative displacements

            assert 1 == 0

            sequence_list.append(sequence)

            if i > 5:
                break
        print("sequence list: ", sequence_list)
        sequences = np.array(sequence_list)
        print("Sequences: ", sequences.shape)

        end = time.time()
        print(f"Time consumed filtering the sequences: {end-start}\n")




if __name__ == "__main__":
    root_folder = "data/datasets/argoverse/motion-forecasting/"
    split = "val"
    obs_len = 20
    pred_len = 30
    distance_threshold = 35
    num_agents_per_obs = 10

    get_sequences_as_array(root_folder=root_folder,split=split,obs_len=obs_len,
                           pred_len=pred_len,distance_threshold=distance_threshold,
                           num_agents_per_obs=num_agents_per_obs)
 