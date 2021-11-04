#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""

Created on Wed Sep 08 16:36:13 2021
@author: Carlos Gómez-Huélamo

http://interaction-dataset.com/details-and-format

Dataloader for the INTERACTION (INTERnational, Adversarial and Cooperative moTION Dataset) dataset

Dataset structure:

Data from 11 locations using drones (DR) or fixed cameras (TC). 12 folders are included in the released "recorded_trackfiles" folder.

For the 11 recording locations, they include three files for each location:

1. High Definition (HD) map (xxx.osm)
2. Recorded vehicle track files (vehicle_tracks_xxx.csv)
3. Recorded pedestrian track files (pedestrian_tracks_xxx.csv)

The maps are with relative coordinate. They did not collect any information with absolute latitude or longitude, 
thus no real latitude or longitude can be found from the maps in the dataset. Also, only motions/trajectories and maps 
are provided, and no raw image is included. However, semantic label map (images) can be generated from the dataset 
as the input of prediction and imitation models with information of motions and maps.

Meta information of the track files 
----------------------------------------
A. Recorded Vehicle Tracks Files (vehicle_tracks_xxx.csv) 

1. track_id: ID of the agent (starts from 1)
2. frame_id: The frames the agent appears in the video (start from 1)
3. timestamp: Represents the time the agent appears in the video (from 100 ms)
4. agent_type: Car, Truck, and so on.
5. x: x-position of the agent at each frame (m)
6. y: y-position of the agent at each frame (m)
7. vx: Vel along x-direction of the agent at each frame (m/s)
8. vy: Vel along y-direction of the agent at each frame (m/s))
9. psi_rad: Yaw angle of the agent at each frame (rad)
10. length: of the agent (m)
11. width: of the agent (m)

B. Recorded Vehicle Tracks Files (person_tracks_xxx.csv) 

1. track_id: ID of the agent (starts from P1)
2. frame_id: The frames the agent appears in the video (start from 1)
3. timestamp: Represents the time the agent appears in the video (from 100 ms)
4. agent_type: Pedestrian/Bycicle.
5. x: x-position of the agent at each frame (m)
6. y: y-position of the agent at each frame (m)
7. vx: Vel along x-direction of the agent at each frame (m/s)
8. vy: Vel along y-direction of the agent at each frame (m/s))

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

sys.path.append("/home/robesafe/tesis/SoPhie/sophie/data_loader")
import dl_aux_functions

def seq_collate_image_aiodrive(data): # id_frame
    """
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, frames, prediction_length, "object_class", "seq_name", 
             "seq_frame", object_id) = batch

        seq:  torch.Size([8, 2]) tensor([[3004.,  720.],
            [4002.,   78.],
            [1014.,  809.],
            [3002.,  139.],
            [4003.,   86.],
            [2007.,  794.],
            [4003.,  340.],
            [4003.,   61.]])

    """
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, idframe_list, vi_path, extension, frame, object_class, objects_id) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] 
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    id_frame = torch.cat(idframe_list, dim=0).permute(2, 0, 1) # seq_len - peds_in_curr_seq - 3
    frames = load_images(list(vi_path), list(frame), extension[0])
    frames = torch.from_numpy(frames).type(torch.float32)
    frames = frames.permute(0, 3, 1, 2)
    object_cls = torch.stack(object_class)
    seq = torch.stack(frame)
    obj_id = torch.stack(objects_id)

    # print("object_class ", object_cls.shape, object_cls)
    # print("seq: ",seq.shape, seq)
    # print("obj_id: ", obj_id.shape, obj_id)

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, id_frame, frames, object_cls, seq, obj_id
    ]

    return tuple(out)

class InteractionDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=0, windows_frames=None, delim='\t', \
        phase='training', split='test', videos_path="", video_extension="png"):
        """
        Args:
        - data_dir: Directory containing dataset files in the format (See above) (.csv)
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(InteractionDataset, self).__init__()
        self.objects_id_dict = {"Car": 0, "Truck": 1, "Ped": 2, "Byc": 3}
        self.dataset_name = "interaction"
        self.videos_path = videos_path
        self.video_extension = video_extension
        self.data_dir = data_dir
        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.delim, self.skip = delim, skip

        all_files, _ = dl_aux_functions.load_list_from_folder(self.data_dir)
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        seq_id_list = []
        frames_list = []
        object_class_id_list = []
        object_id_list = []

        print(">>>>>>>>>>>Split: ", split)

        for path in all_files:
            print_str = 'load %s\r' % path
            sys.stdout.write(print_str)
            sys.stdout.flush()

            _, seq_name, _ = fileparts(path)

            print(">> ", path, seq_name)
            data = dl_aux_functions.read_file(path, delim)
            
            # as testing files only contains past, so add more windows

            if split == 'test':
                min_frame, max_frame = 0, 999
                num_windows = int(max_frame - min_frame + 1 - skip*(self.seq_len - 1))      
                num_windows += (self.pred_len-1)*skip + 1
            else:
                frames = np.unique(data[:, 0]).tolist()
                min_frame, max_frame = frames[0], frames[-1]
                num_windows = int(max_frame - min_frame + 1 - skip*(self.seq_len - 1)) # include all frames for past and future

            # loop through every windows
            for window_index in range(num_windows):
                start_frame = int(window_index + min_frame)
                end_frame = int(start_frame + self.seq_len*skip)        # right-open, not including this frame  
               
                if split=='test':
                    if windows_frames and start_frame not in windows_frames:
                        # print("Continue")
                        continue

                # print("Start frame: ", start_frame)
                    
                frame = start_frame + self.obs_len
                seq_name_int = seqname2int(seq_name)
                if frame > 999:
                    frame -= 1
                seq_frame = np.array([seq_name_int, frame])

                # reduce window during testing, only evaluate every N windows
                # if phase == 'testing':
                #     check_pass = check_eval_windows(start_frame+self.obs_len*skip, self.obs_len*skip, self.pred_len*skip, split=split)
                #     if not check_pass: 
                #         continue

                # get data in current window
                curr_seq_data = []
                for frame in range(start_frame, end_frame, skip):
                    curr_seq_data.append(data[frame == data[:, 0], :])        
                curr_seq_data = np.concatenate(curr_seq_data, axis=0) # frame - id - x - y

                # initialize data

                peds_in_curr_seq_list = []

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # numero de peds en la ventana
                peds_len = peds_in_curr_seq.shape[0]

                num_agents = 32
                num_mini_batches = math.ceil(float(peds_len/32))

                for mini_batch in range(num_mini_batches):
                    if mini_batch == num_mini_batches-1:
                        peds_in_curr_seq_1 = peds_in_curr_seq[num_agents*mini_batch:]
                        dummy = [-1 for i in range(32 - peds_in_curr_seq_1.shape[0])]
                        peds_in_curr_seq_1 = np.concatenate((peds_in_curr_seq_1, dummy))
                        peds_in_curr_seq_list.append(peds_in_curr_seq_1)
                    else:
                        peds_in_curr_seq_1 = peds_in_curr_seq[num_agents*mini_batch:num_agents*(mini_batch+1)]
                        peds_in_curr_seq_list.append(peds_in_curr_seq_1)

                ### crea las esructuras de datos con peds_in_curr_seq de objetos por batch
                # print(">>>>>>>>>>>>>")
                for current_peds in peds_in_curr_seq_list:
                    # print("Current ped: ", current_peds)
                    curr_seq_rel   = np.zeros((len(current_peds), 2, self.seq_len))     # objects x 2 x seq_len
                    curr_seq       = np.zeros((len(current_peds), 2, self.seq_len))
                    curr_loss_mask = np.zeros((len(current_peds)   , self.seq_len))     # objects x seq_len
                    id_frame_list  = np.zeros((len(current_peds), 3, self.seq_len))     # objects x 3 x seq_len
                    object_class_list = np.zeros(len(current_peds))
                    object_class_list.fill(-1)
                    id_frame_list.fill(0)
                    num_peds_considered = 0
                    _non_linear_ped = []

                    # loop through every object in this window
                    # print("current_peds: ", current_peds)
                    for _, ped_id in enumerate(current_peds):
                        if ped_id == -1:
                            num_peds_considered += 1
                            continue

                        object_class = getObjecClass(path)
                        object_class_id = self.objects_id_dict[object_class]
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]      # frame - id - x - y for one of the id of the window, same id
                        pad_front    = int(curr_ped_seq[0, 0] ) - start_frame      # first frame of window       
                        pad_end      = int(curr_ped_seq[-1, 0]) - start_frame + skip # last frame of window
                        assert pad_end % skip == 0, 'error'
                        frame_existing = curr_ped_seq[:, 0].tolist() # frames of windows
                        #print("frame_existing: ", frame_existing, pad_front, pad_end, curr_ped_seq)

                        # pad front and back data to make the trajectory complete
                        if pad_end - pad_front != self.seq_len * skip:
                            
                            # pad end
                            to_be_paded_end = int(self.seq_len - pad_end / skip)
                            pad_end_seq  = np.expand_dims(curr_ped_seq[-1, :], axis=0)
                            pad_end_seq  = np.repeat(pad_end_seq, to_be_paded_end, axis=0)
                            frame_offset = np.zeros((to_be_paded_end, 4), dtype='float32')
                            frame_offset[:, 0] = np.array(range(1, to_be_paded_end+1))
                            pad_end_seq += frame_offset * skip                          # shift first columns for frame
                            curr_ped_seq = np.concatenate((curr_ped_seq, pad_end_seq), axis=0)

                            # pad front
                            to_be_paded_front = int(pad_front / skip)
                            pad_front_seq = np.expand_dims(curr_ped_seq[0, :], axis=0)
                            pad_front_seq = np.repeat(pad_front_seq, to_be_paded_front, axis=0)
                            frame_offset = np.zeros((to_be_paded_front, 4), dtype='float32')
                            frame_offset[:, 0] = np.array(range(-to_be_paded_front, 0))
                            pad_front_seq += frame_offset * skip
                            curr_ped_seq = np.concatenate((pad_front_seq, curr_ped_seq), axis=0)

                            # set pad front and end to correct values
                            pad_front = 0
                            pad_end = self.seq_len * skip

                        # add edge case when the object reappears at a bad frame
                        # in other words, missing intermediate frame
                        if curr_ped_seq.shape[0] != (pad_end - pad_front) / skip:
                            frame_all = list(range(int(curr_ped_seq[0, 0]), int(curr_ped_seq[-1, 0])+skip, skip))     
                            frame_missing, _ = remove_list_from_list(frame_all, curr_ped_seq[:, 0].tolist())

                            # pad all missing frames with zeros
                            pad_seq = np.expand_dims(curr_ped_seq[-1, :], axis=0)
                            pad_seq = np.repeat(pad_seq, len(frame_missing), axis=0)
                            pad_seq.fill(0)
                            pad_seq[:, 0] = np.array(frame_missing)
                            pad_seq[:, 1] = ped_id          # fill ID
                            curr_ped_seq = np.concatenate((curr_ped_seq, pad_seq), axis=0)
                            curr_ped_seq = curr_ped_seq[np.argsort(curr_ped_seq[:, 0])]

                        assert pad_front == 0, 'error'
                        assert pad_end == self.seq_len * skip, 'error'
                        
                        # make sure the seq_len frames are continuous, no jumping frames
                        start_frame_now = int(curr_ped_seq[0, 0])
                        if curr_ped_seq[-1, 0] != start_frame_now + (self.seq_len-1)*skip:
                            num_peds_considered += 1
                            continue

                        # make sure that past data has at least one frame
                        past_frame_list = [*range(start_frame_now, start_frame_now + self.obs_len * skip, skip)]
                        common = find_unique_common_from_lists(past_frame_list, frame_existing, only_com=True)
                        #print("common ", common)
                        if len(common) == 0:
                            num_peds_considered += 1
                            continue

                        # make sure that future GT data has at least one frame
                        if phase != 'testing':
                            gt_frame_list = [*range(start_frame_now + self.obs_len*skip, start_frame_now + self.seq_len*skip, skip)]
                            common = find_unique_common_from_lists(gt_frame_list, frame_existing, only_com=True)
                            if len(common) == 0: 
                                num_peds_considered += 1
                                continue

                        # only keep the state
                        cache_tmp = np.transpose(curr_ped_seq[:, :2])       # 2xseq_len | [0,:] list of frames | [1,:] id
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])    # 2 x seq_len | [0,:] x | [1,:] y

                        # print("cache_tmp: ", cache_tmp, cache_tmp.shape)
                        # print("curr_ped_seq: ", curr_ped_seq, curr_ped_seq.shape)

                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        _idx = num_peds_considered
                        curr_seq[_idx, :, :] = curr_ped_seq     
                        curr_seq_rel[_idx, :, :] = rel_curr_ped_seq

                        # record seqname, frame and ID information 20 - 3 - x
                        id_frame_list[_idx, :2, :] = cache_tmp
                        id_frame_list[_idx, 2, :] = seq_name_int # img_id - ped_id - seqname2int
                        
                        # Linear vs Non-Linear Trajectory, only fit for the future part not past part

                        if phase != 'testing':
                            _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))     

                        # add mask onto padded dummay data
                        frame_exist_index = np.array([frame_tmp - start_frame_now for frame_tmp in frame_existing])
                        frame_exist_index = (frame_exist_index / skip).astype('uint8')
                        curr_loss_mask[_idx, frame_exist_index] = 1

                        # object id
                        object_class_list[num_peds_considered] = object_class_id

                        num_peds_considered += 1
                        #print("b")
                    
                    #print("num_peds_considered ", num_peds_considered)
                    #num_peds_considered = 32
                    if num_peds_considered > min_ped:
                        if len(_non_linear_ped) != num_peds_considered:
                            dummy = [-1 for i in range(num_peds_considered - len(_non_linear_ped))]
                            _non_linear_ped = _non_linear_ped + dummy
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                        seq_id_list.append(id_frame_list[:num_peds_considered])
                        frames_list.append(seq_frame)
                        object_class_id_list.append(object_class_list)
                        object_id_list.append(id_frame_list[:num_peds_considered][:,1,0])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)             # objects x 2 x seq_len
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        seq_id_list = np.concatenate(seq_id_list, axis=0)
        frames_list = np.asarray(frames_list)
        object_class_id_list = np.asarray(object_class_id_list)
        object_id_list = np.asarray(object_id_list)
        #print("seq_list: ", seq_list.shape)
        #print("frames_list: ", frames_list.shape)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.seq_id_list = torch.from_numpy(seq_id_list).type(torch.float)
        self.frames_list = torch.from_numpy(frames_list).type(torch.float)
        self.object_class_id_list = torch.from_numpy(object_class_id_list).type(torch.float)
        self.object_id_list = torch.from_numpy(object_id_list).type(torch.float)
        # print("ID List: ", self.object_id_list)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.seq_id_list[start:end, :], self.videos_path, self.video_extension, self.frames_list[index, :],
            self.object_class_id_list[index], self.object_id_list[index]
        ]

        return out



