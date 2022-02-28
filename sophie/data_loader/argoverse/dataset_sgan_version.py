import logging
import os
import math
import csv
import time
import pdb
import copy
import glob2
import glob
import multiprocessing
from numpy.random import default_rng

from sklearn import linear_model

import cv2
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.dummy import Pool

import torch
from torch.utils.data import Dataset

from numba import jit

from argoverse.map_representation.map_api import ArgoverseMap
import sophie.data_loader.argoverse.map_utils as map_utils

frames_path = None
avm = ArgoverseMap()
dist_around = 40
dist_rasterized_map = [-dist_around, dist_around, -dist_around, dist_around]

def isstring(string_test):
    """
    """
    return isinstance(string_test, str)

def safe_path(input_path):
    """
    """
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

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

def load_images(num_seq, obs_seq_data, city_id, ego_origin, dist_rasterized_map, 
    num_agents_per_obs, object_class_id_list,debug_images=False):
    """
    Get the corresponding rasterized map
    """

    # print("LOAD IMAGES")

    batch_size = len(object_class_id_list)
    frames_list = []
    # pdb.set_trace()

    # rasterized_start = time.time()
    t0_idx = 0
    for i in range(batch_size):
        
        curr_num_seq = int(num_seq[i].cpu().data.numpy())
        object_class_id = object_class_id_list[i].cpu().data.numpy()

        # print("obj id list: ", obj_id_list)
        t1_idx = len(object_class_id_list[i]) + t0_idx
        if i < batch_size - 1:
            curr_obs_seq_data = obs_seq_data[:,t0_idx:t1_idx,:]
        else:
            curr_obs_seq_data = obs_seq_data[:,t0_idx:,:]

        curr_city = city_id[i]
        if curr_city == 0:
            city_name = "PIT"
        else:
            city_name = "MIA"

        curr_ego_origin = ego_origin[i].reshape(1,-1)
        start = time.time()

        curr_obs_seq_data = curr_obs_seq_data.reshape(-1,2) # Past_Observations x Num_Agents x 2 -> (Past_Observations * Num_agents) x 2 
                                                            # (required by map_utils)
        fig = map_utils.map_generator(
            curr_obs_seq_data, curr_ego_origin, dist_rasterized_map, avm, city_name,
            (object_class_id, num_agents_per_obs), show=False, smoothen=True
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

            # assert 1 == 0

        plt.close("all")
        end = time.time()
        frames_list.append(img)
        t0_idx = t1_idx
        # print(f"Time consumed by map render: {end-start}")

    # rasterized_end = time.time()
    # print(f"Time consumed by rasterized image: {rasterized_end-rasterized_start}")

    frames_arr = np.array(frames_list)
    return frames_arr

def seq_collate(data): # 2.58 seconds - batch 8
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
     object_id_list, city_id, ego_vehicle_origin, num_seq_list, norm) = zip(*data)

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
    # frames = load_images(num_seq_list, obs_traj_rel, city_id, ego_vehicle_origin,    # Return batch_size x 600 x 600 x 3
    #                      dist_rasterized_map, num_agents_per_obs, object_class_id_list, debug_images=False)
    frames = np.random.randn(1,1,1,1)
    end = time.time()
    # print(f"Time consumed by load_images function: {end-start}\n")

    frames = torch.from_numpy(frames).type(torch.float32)
    frames = frames.permute(0, 3, 1, 2)
    # pdb.set_trace()
    object_cls = torch.cat(object_class_id_list, dim=0)
    obj_id = torch.cat(object_id_list, dim=0)
    ego_vehicle_origin = torch.stack(ego_vehicle_origin)
    num_seq_list = torch.stack(num_seq_list)
    norm = torch.stack(norm)

    out = [obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
           loss_mask, seq_start_end, frames, object_cls, obj_id, ego_vehicle_origin, num_seq_list, norm]

    end = time.time()
    # print(f"Time consumed by seq_collate function: {end-start}\n")

    return tuple(out)

# time 1 csv -> 0.0104s | 200000 csv -> 34m
def read_file(_path):
    data = csv.DictReader(open(_path))
    aux = []
    id_list = []
    for row in data:
        values = list(row.values())
        # object type 
        values[2] = 0 if values[2] == "AV" else 1 if values[2] == "AGENT" else 2
        # city
        values[-1] = 0 if values[-1] == "PIT" else 1
        # id
        id_list.append(values[1])
        # numpy_sequence
        aux.append(values)
    id_list, id_idx = np.unique(id_list, return_inverse=True)
    data = np.array(aux)
    data[:, 1] = id_idx
    return data.astype(np.float64)

# @jit(nopython=True)
def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

# @jit(nopython=True)
def process_window_sequence(idx, frame_data, frames, seq_len, pred_len, \
    threshold, file_id, split, skip=1):
    """
    frame_data array (n, 6):
        - timestamp (int)
        - id (int) -> previously need to be converted. Original data is string
        - type (int) -> need to be converted from string to int
        - x (float) -> x position
        - y (float) -> y position
        - city_name (int)
    seq_len (int)
    skip (int)
    pred_len (int)
    threshold (float)
    """
    # try:
    curr_seq_data = np.concatenate( # 90, 6
        frame_data[idx:idx + seq_len], axis=0)
    peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # 13
    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, # 13, 2, 50 (sel.seq_len)
                                seq_len))
    curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len)) # 13, 2, 50
    curr_loss_mask = np.zeros((len(peds_in_curr_seq), # 13, 50
                                seq_len))

    object_class_list = np.zeros(len(peds_in_curr_seq)) # 13
    id_frame_list  = np.zeros((len(peds_in_curr_seq), 3, seq_len))
    num_objs_considered = 0
    _non_linear_obj = []
    ego_origin = []
    city_id = curr_seq_data[0,5]
    for _, ped_id in enumerate(peds_in_curr_seq):
        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == # 2, 6
                                        ped_id, :]
        # curr_ped_seq = np.around(curr_ped_seq, decimals=4)
        #################################################################################
        ## test
        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
        #################################################################################
        if (pad_end - pad_front != seq_len) or (curr_ped_seq.shape[0] != seq_len): # -> si es menor a seq_len, fuera
            continue
        ### ego
        if curr_ped_seq[0,2] == 1:
            ego_vehicle = curr_ped_seq[19, 3:5]
            ego_origin.append(ego_vehicle)
        # object class id
        object_class_list[num_objs_considered] = curr_ped_seq[0,2] # ?
        # Record seqname, frame and ID information
        cache_tmp = np.transpose(curr_ped_seq[:,:2])
        id_frame_list[num_objs_considered, :2, :] = cache_tmp
        id_frame_list[num_objs_considered,  2, :] = file_id
        # get  x- y data
        curr_ped_seq = np.transpose(curr_ped_seq[:, 3:5]) # 2, 16
        curr_ped_seq = curr_ped_seq
        # Make coordinates relative
        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape) # 2, 16
        rel_curr_ped_seq[:, 1:] = \
            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
        _idx = num_objs_considered
        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
        # Linear vs Non-Linear Trajectory
        if split != 'test':
            _non_linear_obj.append(
                poly_fit(curr_ped_seq, pred_len, threshold))
        curr_loss_mask[_idx, pad_front:pad_end] = 1

        # add num_objs_considered
        num_objs_considered += 1
    # except Exception as e:
    #     print("Error ", e)
    #     pdb.set_trace()
    return num_objs_considered, _non_linear_obj, curr_loss_mask, \
        curr_seq, curr_seq_rel, id_frame_list, object_class_list, city_id, ego_origin


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    return lst

def load_sequences_thread(files):
    sequences = []
    for i, path in enumerate(files):
        file_id = int(path.split("/")[-1].split(".")[0])
        sequences.append([read_file(path), file_id])
    return sequences

class ArgoverseMotionForecastingDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, dataset_name, root_folder, obs_len=20, pred_len=30, skip=1, threshold=0.002, distance_threshold=30,
                 min_objs=0, windows_frames=None, split='train', num_agents_per_obs=10, split_percentage=0.1, shuffle=False):
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
        self.shuffle = shuffle
        self.min_ped = 2
        self.ego_vehicle_origin = []

        folder = root_folder + split + "/data/"
        files, num_files = load_list_from_folder(folder)

        if self.shuffle:
            rng = default_rng()
            indeces = rng.choice(num_files, size=int(num_files*split_percentage), replace=False)
            files = np.take(files, indeces, axis=0)
        else:
            files = files[:int(num_files*split_percentage)]

        num_objs_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_obj = []
        seq_id_list = []
        object_class_id_list = []
        object_id_list = []
        num_seq_list = []
        self.city_ids = []

        min_disp_rel = []
        max_disp_rel = []

        print("Start Dataset")
        t0 = time.time()
        for i, path in enumerate(files):
            file_id = int(path.split("/")[-1].split(".")[0])
            # print(f"File {i}/{len(files)}")
            num_seq_list.append(file_id)
            data = read_file(path) # 4946, 4 | biwi_hotel_train
            frames = np.unique(data[:, 0]).tolist() # 934
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :]) # save info for each frame
            num_sequences = int( # 919
                math.ceil((len(frames) - self.seq_len + 1) / skip)) # (934 - 16 + 1) / 1 
            idx = 0
            num_objs_considered, _non_linear_obj, curr_loss_mask, curr_seq, \
                curr_seq_rel, id_frame_list, object_class_list, city_id, ego_origin = \
                process_window_sequence(idx, frame_data, frames, \
                    self.seq_len, self.pred_len, threshold, file_id, self.split)

            # agent_idx = np.where(object_class_list == self.objects_id_dict['AGENT'])[0]
            # agent_seq = curr_seq[agent_idx,:,:]

            # ransac = linear_model.RANSACRegressor(max_trials=100,min_samples=40)
            # agent_x = agent_seq[0,0,:].reshape(-1,1)
            # agent_y = agent_seq[0,1,:].reshape(-1,1)
            # ransac.fit(agent_x,agent_y)

            # line_x = np.arange(agent_x.min(), agent_x.max())[:, np.newaxis]
            # line_y_ransac = ransac.predict(line_x)

            # plt.scatter(agent_x,agent_y, color='yellowgreen', marker='.',
            #             label='Inliers')
            # plt.plot(line_x, line_y_ransac, color='cornflowerblue', linewidth=1,
            #         label='RANSAC regressor')
            # plt.legend(loc='lower right')
            # plt.xlabel("Input")
            # plt.ylabel("Response")
            # plt.show()

            # pdb.set_trace()
            # if (curr_seq_rel.min() < -3.5) or (curr_seq_rel.max() > 3.5):

            # min_disp_rel.append(curr_seq_rel.min())
            # max_disp_rel.append(curr_seq_rel.max())

            # continue
            
            # pdb.set_trace()

            if num_objs_considered >= self.min_ped:
                non_linear_obj += _non_linear_obj
                num_objs_in_seq.append(num_objs_considered)
                loss_mask_list.append(curr_loss_mask[:num_objs_considered])
                seq_list.append(curr_seq[:num_objs_considered]) # elimina dummies
                seq_list_rel.append(curr_seq_rel[:num_objs_considered])
                ###################################################################
                seq_id_list.append(id_frame_list[:num_objs_considered]) # (timestamp, id, file_id)
                object_class_id_list.append(object_class_list[:num_objs_considered]) # obj_class (-1 0 1 2 2 2 2 ...)
                object_id_list.append(id_frame_list[:num_objs_considered,1,0])
                ###################################################################
                self.city_ids.append(city_id)
                self.ego_vehicle_origin.append(ego_origin)

        # disp_hist = min_disp_rel + max_disp_rel
        # disp_hist = np.array(disp_hist)

        # n, bins, patches = plt.hist(disp_hist, bins=40)
        # plt.show()

        # pdb.set_trace()
        print("Dataset time: ", time.time() - t0)
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # Objects x 2 x seq_len
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_obj = np.asarray(non_linear_obj)
        seq_id_list = np.concatenate(seq_id_list, axis=0)
        object_class_id_list = np.concatenate(object_class_id_list, axis=0)
        object_id_list = np.concatenate(object_id_list)
        num_seq_list = np.concatenate([num_seq_list])
        self.ego_vehicle_origin = np.asarray(self.ego_vehicle_origin)

        ## normalize abs and relative data
        abs_norm = (seq_list.min(), seq_list.max())
        # seq_list = (seq_list - seq_list.min()) / (seq_list.max() - seq_list.min())

        rel_norm = (seq_list_rel.min(), seq_list_rel.max())
        # seq_list_rel = (seq_list_rel - seq_list_rel.min()) / (seq_list_rel.max() - seq_list_rel.min())
        norm = (abs_norm, rel_norm)

        ## create torch data
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
        self.norm = torch.from_numpy(np.array(norm))
        

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        # print("self.object_class_id_list[start:end] ", self.object_class_id_list[start:end])
        out = [
                self.obs_traj[start:end, :, :], self.pred_traj_gt[start:end, :, :],
                self.obs_traj_rel[start:end, :, :], self.pred_traj_gt_rel[start:end, :, :],
                self.non_linear_obj[start:end], self.loss_mask[start:end, :],
                self.seq_id_list[start:end, :, :], self.object_class_id_list[start:end], 
                self.object_id_list[start:end], self.city_ids[index], self.ego_vehicle_origin[index,:,:],
                self.num_seq_list[index], self.norm
              ] 
        return out
