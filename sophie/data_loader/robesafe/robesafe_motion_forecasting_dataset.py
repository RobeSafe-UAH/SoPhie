#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Fri Sep 24 13:59:17 2021
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
     non_linear_obj, loss_mask, seq_id_list, frames_path, 
     frames_extension, frames_list, object_class_id_list, object_id_list) = zip(*data)

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

# TODO: DELETE THIS, these functions must be in another file as ad-hoc dataset functions

def seqname2int(seqname): # Ex: Town01_seq0001.txt -> 1*1000 + 1 
    city_id, seq_id = seqname.split('_')
    city_id = int(city_id[4:6])
    seq_id = int(seq_id[3:])
    final_id = city_id * 1000 + seq_id

    return final_id

def check_eval_windows(start_pred, obs_len, pred_len, phase='test'):
    # start_pred:       the frame index starts to predict in this window, e.g., seq of 0-9 -> 10-19 has start frame at 10
    # pred_len:         the number of frames to predict 
    # phase:            train, val, test

    if phase == 'test':
        reserve_interval = 50
        pred_len = 50
        obs_len  = 50
    else:
        reserve_interval = 0
    check = (start_pred - obs_len) % (obs_len + pred_len + reserve_interval) == 0

    return check

def getObjecClass(url_dataset):
    url_list = url_dataset.split("/")
    objectClass = ""
    for url_id in url_list:
        if "aiodrive_" in url_id:
            objectClass = url_id
    objectClass = objectClass.split("_")[-1]
    return objectClass

# End TODO: DELETE THIS, these functions must be in another file as ad-hoc dataset functions   

class RobeSafeMotionForecastingDataset(Dataset):
    """
    Dataloader for generic trajectory forecasting datasets
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
                 min_objs=0, windows_frames=None, phase='train', delim='\t', frames_path="", frames_extension='png', num_agents=32):
        """
        - data_dir: Directory containing dataset files in the format 
          <frame_id> <object_id> <x> <y>
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
        - num_agents: Number of agents to be considered in a single forward. If there are less than num_agents,
          dummy variables are used to predict. If there are more, it is distributed in total_agents % num_agents
          forwards but using the same physical information and frame index
        """
        super(RobeSafeMotionForecastingDataset, self).__init__()
        self.dataset_name = "robesafe_motion_forecasting_dataset"
        self.objects_id_dict = {"Car": 0, "Cyc": 1, "Mot": 2, "Ped": 3} # TODO: Get this by argument
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



                








    




