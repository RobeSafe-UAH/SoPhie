# It is necessary to add the image to the dataset
import os
import numpy as np
import math
import cv2

import torch
from torch.utils.data.dataset import Dataset

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

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
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)

def seq_collate_image(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, frames) = zip(*data)

    # (1,600,600,3)
    frames_arr = tuple([torch.from_numpy(frame).type(torch.float32).unsqueeze(0) for frame in frames])
    print("frames_arr: ", frames_arr[0].shape)

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
    frames_t = torch.cat(frames_arr).permute(0, 3, 1, 2)

    print("frames_t: ", type(frames_t), frames_t.shape)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, frames_t
    ]

    return tuple(out)

def read_file(_path, delim='tab'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
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
    first_token = path.split("/")[-1][0]
    return True if first_token == "." else False

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

class EthUcyDataset(Dataset):
    
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, #pred_len=12 
        min_ped=1, delim='\t', img_shape=(600,600), videos_path="", video_extension="avi"
    ):
        super(EthUcyDataset, self).__init__()

        self.threshold = threshold
        self.min_ped = min_ped
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.img_shape = img_shape
        self.videos_path = videos_path
        self.video_extension = video_extension
        self.prepare_dataset()

    def prepare_dataset(self):
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        frames_list = []
        frame_dict_dataset = []

        for path in all_files:
            # read file with data file with
            # structure: frame - x -y- z
            #print("path: ", path)
            if ignore_file(path):
                continue
            dataset_name = self.get_dataset_name(path)
            #print("dataset_name: ", dataset_name)
            #data = read_file(self.videos_path + "/" + dataset_name, self.delim)
            data = read_file(path, self.delim)
            # obtain frames from all the data and
            # create data structure oredered by frame
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / self.skip)) #>? why add 1

            frames_dts = []

            #print("num_sequences ", len(frames), self.seq_len, self.skip, num_sequences)
            
            for idx in range(0, num_sequences * self.skip + 1, self.skip):

                ## secuencia
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)

                ## agentes en secuencia
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                #print("peds_in_curr_seq: ", peds_in_curr_seq)
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, # numero de agentes en escena, xy, longitud secuencia
                                         self.seq_len))
                #print("curr_seq_rel ", curr_seq_rel.shape)
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                frame_idx_list = []

                for _, ped_id in enumerate(peds_in_curr_seq):

                    ## get [frame id x y] of agent
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # get index of frame for current agent. start and end
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    #print("pad: ", pad_front, pad_end)
                    if pad_end - pad_front != self.seq_len:
                        continue

                    frame_idx = frames.index(curr_ped_seq[-1, 0]) - self.obs_len + 1
                    frame_idx_list.append(frame_idx)
                    
                    ## get points of agents (2,self.seq_len)
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    #print("curr_ped_seq: ", curr_ped_seq)

                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    #print("rel_curr_ped_seq: ", rel_curr_ped_seq)
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, self.pred_len, self.threshold))

                    #print("_non_linear_ped: ", _non_linear_ped)
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    #print("curr_loss_mask: ", curr_loss_mask, curr_loss_mask.shape)
                    num_peds_considered += 1

                #print("=======================" , num_peds_considered)
                if num_peds_considered > self.min_ped:
                    num_peds_considered = 32
                    if len(_non_linear_ped) < num_peds_considered:
                        len_nlp = len(_non_linear_ped)
                        dumm = [0 for _ in range(num_peds_considered-len_nlp)]
                        _non_linear_ped = _non_linear_ped + dumm

                    non_linear_ped += _non_linear_ped
                    #print("REEEE ", _non_linear_ped)
                    num_peds_in_seq.append(num_peds_considered)
                    #print("REEEE ", num_peds_considered)
                    if curr_loss_mask[:num_peds_considered].shape[0] != num_peds_considered:
                        (f_dim, s_dim) = curr_loss_mask[:num_peds_considered].shape
                        dumm = np.zeros((num_peds_considered-f_dim, s_dim))
                        curr_loss_mask_ped = curr_loss_mask[:num_peds_considered]
                        curr_loss_mask_ped_final = np.concatenate((curr_loss_mask_ped, dumm), axis=0)

                    loss_mask_list.append(curr_loss_mask_ped_final)
                    #print("REEEE ", curr_loss_mask_ped_final.shape)

                    curr_seq_ped = curr_seq[:num_peds_considered]
                    if curr_seq_ped.shape[0] != num_peds_considered:
                        (f_dim, s_dim, t_dim) = curr_seq_ped.shape
                        dumm = np.zeros((num_peds_considered-f_dim, s_dim, t_dim))
                        curr_seq_ped_final = np.concatenate((curr_seq_ped, dumm), axis=0)
                    seq_list.append(curr_seq_ped_final)
                    #print("REEEE ", curr_seq_ped_final.shape)

                    curr_seq_rel_ped = curr_seq_rel[:num_peds_considered]
                    if curr_seq_rel_ped.shape[0] != num_peds_considered:
                        (f_dim, s_dim, t_dim) = curr_seq_rel_ped.shape
                        dumm = np.zeros((num_peds_considered-f_dim, s_dim, t_dim))
                        curr_seq_ped_final = np.concatenate((curr_seq_rel_ped, dumm), axis=0)
                    seq_list_rel.append(curr_seq_ped_final)
                    #print("REEEE ", curr_seq_ped_final.shape)

                    #print("frame_idx_list: ", np.unique(frame_idx_list))
                    frames_dts.append(np.unique(frame_idx_list)[0])
                    #assert 1 == 0, "aiiieee"

            frame_dict_dataset.append({dataset_name: frames_dts})
        
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0).astype(np.float32)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0).astype(np.float32)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0).astype(np.float32)
        non_linear_ped = np.asarray(non_linear_ped).astype(np.float32)

        ### get frames from dataset
        iteration = 0
        for data in frame_dict_dataset:
            key, value = list(data.keys()), list(data.values())
            frame_list_im = self.get_frames(
                os.path.join(self.videos_path, ".".join([key[0], self.video_extension])),
                self.img_shape, 
                value[0]
            )
            frame_list_im = np.concatenate(frame_list_im, axis=0)
            if iteration == 0:
                frames_list = frame_list_im
                iteration += 1
                continue
            frames_list = np.concatenate((frames_list, frame_list_im), axis=0)
        
        print(">>> ", frames_list.shape)

        #assert 1== 0, "re"

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float32) # [53472, 2, 8]
        print("self.obs_traj: ", self.obs_traj.shape)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float32) # [53472, 2, 8]
        print("self.pred_traj: ", self.pred_traj.shape)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float32) # [53472, 2, 8]
        print("obs_traj_rel: ", self.obs_traj_rel.shape)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float32) # [53472, 2, 8]
        print("pred_traj_rel: ", self.pred_traj_rel.shape)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float32) # [53472, 16]
        print("loss_mask: ", self.loss_mask.shape)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float32) # [53472]
        print("non_linear_ped: ", self.non_linear_ped.shape)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() # [1672]
        print("cum_start_idx: ", len(cum_start_idx)) 
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:]) # [1671]
        ]
        print("seq_start_end: ", len(self.seq_start_end))
        # self.frames = torch.from_numpy(
        #     frames_list).type(torch.float32)
        self.frames = frames_list # (1671, 600, 600, 3)
        print("frames: ", self.frames.shape)

    def get_frames(self, path, new_shape, frames):
        cap = cv2.VideoCapture(path) 
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_list = []
        frame_counter = 0
        while (num_frames > 0):
            _, frame = cap.read()
            num_frames -= 1
            frame_counter += 1
            if frame_counter not in frames:
                continue
            re_frame = cv2.resize(frame, new_shape)
            frames_list.append(np.expand_dims(re_frame, axis=0))
        cap.release()
        return frames_list

    def get_dataset_name(self, path):
        dts_name = path.split("/")[-1].split(".")[0].split("_")[0:-1]
        dts_name = "_".join(dts_name)
        return dts_name


    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
            ,self.frames[idx, :]
        ]
        return out

    def prepare_batch(self, batch):
        return 1