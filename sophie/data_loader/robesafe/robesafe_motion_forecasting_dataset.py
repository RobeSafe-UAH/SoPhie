#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""

Created on Fri Sep 24 13:59:17 2021
@author: Carlos Gómez-Huélamo

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

from torch.utils.data import Dataset

np.set_printoptions(precision=3, suppress=True)

## File managements functions

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
    return isistance(string_test, str)

def find_unique_common_from_lists(input_list1, input_list2, only_com=False):
    """
    """
    input_list1 = safe_list(input_list1)
    input_list2 = safe_list(input_list2)

    common_list = list(set(input_list1).intersection(input_list2))

    if only_com:
        return common_list
    else: # Find index of the common elements
        index_list1 = []
        for index in range(len(input_list1)):
            item = input_list1[index]
            if item in common_list:
                index_list1.append(index)

        index_list2 = []
        for index in range(len(input_list2)):
            item = input_list2[index]
            if item in common_list:
                index_list2.append(index)
        return common_list, index_list1, index_list2

def remove_list_from_list(input_list, list_toremove_src):
    """
    """
    list_remained = safe_list(input_list)
    list_toremove = safe_list(list_toremove_src)
    list_removed = []

    for item in list_toremove:
        try:
            list_remained.remove(item)
            list_removed.append(item)
        except ValueError:
            if warning: 
                print("Item to remove is not in the list. Remove operation is not done")
    return list_remained, list_removed

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

def fileparts(input_path):
    good_path = safe_path(input_path)
    if len(good_path) == 0: 
        return('','','')
    if good_path[-1] == '/':
        if len(good_path) > 1: 
            return (good_path[:-1],'','') # Ignore the final '/'
        else:
            return (good_path,'','') # Ignore the final '/'

    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]

    return (directory, filename, ext)

def get_folder_name(video_path, seq_name):
    """
    """

    print("REQUIRED HERE?")

def load_images(video_path, frames):
    """
    """

    print("Aquí tenemos mapa rasterizado para una secuencia, pero solo uno por secuencia????")

def read_file(_path, delim='tab'):
    """
    """
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def ignore_file(path):
    """
    """
    first_token = path.split("/")[-1][0]
    return True if firsk_token == "." else False

## End File management functions

## Motion Prediction functions

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

def seq_collate(data):
    """
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_obj, loss_mask, frames, "object_class", 
     "seq_frame", "object_id") = batch
    """

class RobeSafeMotionForecastingDataset(Dataset):
    """
    Dataloader for generic trajectory forecasting datasets
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
                 min_obj=1, delim='\t', frames_path=""):
        """
        - data_dir: Directory containing dataset files in the format 
          <frame_id> <object_id> <x> <y>
        - obs_len: Number of observed frames in prior (input) trajectories
        - pred_len: Number of predicted frames in posterior (output) trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using 
          a linear predictor
        - min_obj: Minimum number of objects that should be in a sequence
        - delim: Delimiter in the dataset files
        - frames_path: Path to physical information (in this case HDMaps in .npy format)
        """
        super(RobeSafeMotionForecastingDataset, self).__init__()
        self.dataset_name = "robesafe_motion_forecasting_dataset"
        self.data_dir = data_dir
        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.skip, self.delim = skip, delim
        self.frames_path = frames_path

        all_files, _ = load_list_from_folder(self.data_dir)
        num_objs_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_obj = []
        seq_id_list = []
        frames_list = []
        object_class_id_list = []
        object_id_list = []

        for path in all_files:
            print_str = 'load %s\r' % path
            sys.stdout.write(print_str)
            sys.stdout.flush()

            _, seq_name, _ = fileparts(path)

            print(">> ", path, seq_name)
            data = read_file(path, delim)







    




