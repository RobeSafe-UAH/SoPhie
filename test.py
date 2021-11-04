import os
import numpy as np
import copy
import yaml
import cv2
import glob, glob2
import time
import json
import pandas as pd
import math

from types import SimpleNamespace
from sophie.models import SoPhieDiscriminator, SoPhieGenerator
from sophie.modules.layers import MLP
from sophie.modules.backbones import VisualExtractor, JointExtractor
from sophie.modules.encoders import Encoder
from sophie.modules.classifiers import Classifier
from sophie.data_loader.ethucy.dataset import read_file, EthUcyDataset, seq_collate, seq_collate_image
from sophie.data_loader.aiodrive.dataset import AioDriveDataset, seq_collate_image_aiodrive
from sophie.modules.decoders import Decoder
from sophie.modules.attention import SATAttentionModule

from prodict import Prodict
import imageio

import torch
from torch import nn
from torch import rand
from torch.utils.data import DataLoader

# Global variables

## CUDA device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Configuration file

with open(r'./configs/sophie_argoverse.yml') as config_file:
    config_file = yaml.safe_load(config_file)
    config_file = Prodict.from_dict(config_file)

## Batch size

batch_size = config_file.dataset.batch_size # Dataloader batch

## Image

im_channels = 3
im_width = 600
im_height = 600

## Processed image

vgg_channels = 512
vgg_channels_width = 18
vgg_channels_height = 18

## Trajectories

po = 20 # Past Observations
fo = 30 # Future Observations
na = 10 # Number of Agents 
tfd = 2 # Trajectory Features Dimension (x,y per point)

## Decoder features

decoder_dim_features = 128

# Functions
    
## Read data

def test_read_file():
    data = read_file("./data/datasets/eth/test/biwi_eth.txt", "tab")
    print("data: ", data)
    frames = np.unique(data[:, 0]).tolist()
    print("frames: ", frames)

def read_video(path, new_shape):
    cap = cv2.VideoCapture(path) 
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_list = []
    while (num_frames > 0):
        _, frame = cap.read()
        print("_ ", _)
        num_frames = num_frames - 1
        re_frame = cv2.resize(frame, new_shape)
        frames_list.append(re_frame)
    cap.release()
    return frames_list

def test_dataLoader_img():
    data = EthUcyDataset("./data/datasets/zara1/train", videos_path="./data/datasets/videos/")
    print(data)
    batch_size = 64
    loader_num_workers = 4
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate_image)

    print("loader: ", loader)
    print("device: ", device)
    t0 = time.time()
    for batch in loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end, frames) = batch

        print("> ", obs_traj.shape, pred_traj_gt.shape, frames.shape)
        # t1 = time.time()
        # while(t1 - t0 < 120):
        #     print(t1-t0)
        #     t1 = time.time()
        #assert 1 == 0, "aiie"

## Extractors

def test_visual_extractor():
    # opt = {
    #     "vgg_type": 19,
    #     "batch_norm": False,
    #     "pretrained": True,
    #     "features": True
    # }

    # vgg_19 = VisualExtractor("vgg19", opt).to(device)

    vgg_19 = VisualExtractor("vgg19", config_file.sophie.generator.visual_extractor.vgg).to(device)
    image_test = rand(batch_size,3,im_height,im_width).to(device) # batch, channel, H, W
    print(">>> ", vgg_19(image_test).shape) # batch, 512, 18, 18

def test_joint_extractor():
    initial_trajectory = 10 * np.random.randn(po, na*batch_size, tfd)
    initial_trajectory = torch.from_numpy(initial_trajectory).to(device).float()

    # opt = {
    #     "encoder": {
    #         "num_layers": 1,
    #         "hidden_dim": 32,
    #         "emb_dim": 2,
    #         "mlp_config": {
    #             "dim_list": [2, 32],
    #             "activation": 'relu',
    #             "batch_norm": False,
    #             "dropout": 0
    #         },
    #         "dropout": 0,
    #     }
    # }

    # opt = Prodict.from_dict(opt)

    # joint_extractor = JointExtractor("encoder_sort", opt).to(device)

    joint_extractor = JointExtractor("encoder_sort", config_file.sophie.generator.joint_extractor.config).to(device)
    joint_features, _ = joint_extractor(initial_trajectory)

    print("Joint Features: ", joint_features.shape)

## Attention modules 

def test_physical_attention_model():
    """
    """
    print("Physical attention model")
    # Feature 1 (Visual Extractor)

    visual_extractor_features = 10 * np.random.randn(batch_size, vgg_channels, vgg_channels_width, vgg_channels_height)
    visual_extractor_features = torch.from_numpy(visual_extractor_features).to(device).float()

    # Hidden Decoder Features

    predicted_trajectory = 10 * np.random.randn(1, na*batch_size, decoder_dim_features)
    predicted_trajectory = torch.from_numpy(predicted_trajectory).to(device).float()

    # print("Physical attention config: ", config_file.sophie.generator.physical_attention)
    physical_attention_module = SATAttentionModule(config_file.sophie.generator.physical_attention).to(device)
    # print("Physical Attention Module: ", physical_attention_module)

    alpha, context_vector = physical_attention_module.forward(visual_extractor_features, predicted_trajectory)

    # print("Alpha: ", alpha.shape)
    print("Physical context vector: ", context_vector.shape)

    return context_vector

def test_social_attention_model():
    """
    """
    print("Social attention model")
    # Feature 1 (Joint Extractor)

    hidden_dim = 32 # Features dimension
 
    joint_extractor_features = 10 * np.random.randn(po, na*batch_size, hidden_dim)
    joint_extractor_features = torch.from_numpy(joint_extractor_features).to(device).float()

    # Hidden Decoder Features

    hidden_decoder_features = 10 * np.random.randn(1, na*batch_size, decoder_dim_features)
    hidden_decoder_features = torch.from_numpy(hidden_decoder_features).to(device).float()

    # print("Social attention config: ", config_file.sophie.generator.social_attention)
    social_attention_module = SATAttentionModule(config_file.sophie.generator.social_attention).to(device)
    # print("Social Attention Module: ", social_attention_module)

    # print("Joint extractor features: ", joint_extractor_features.shape)
    # print("Hidden decoder features: ", hidden_decoder_features.shape)
    alpha, context_vector = social_attention_module.forward(joint_extractor_features, hidden_decoder_features)

    # print("Alpha: ", alpha.shape)
    print("Social context vector: ", context_vector.shape)

    return context_vector

def test_concat_features():
    """
    """
    
    # physical_context_vector = test_physical_attention_model().contiguous()
    # social_context_vector = test_social_attention_model().contiguous()
    physical_context_vector = test_physical_attention_model().contiguous()
    print("\n")
    social_context_vector = test_social_attention_model().contiguous()

    attention_features = torch.matmul(physical_context_vector, social_context_vector.t())
    # attention_features = torch.cat((physical_context_vector, social_context_vector), 0).to(device)
    print("\n")
    print("Attention features: ", attention_features.shape)
    generator = SoPhieGenerator(config_file.sophie.generator)

    generator.build()
    generator.to(device)

    shape_features = attention_features.shape
    noise = generator.create_white_noise(
        generator.config.noise.noise_type,
        shape_features
    )

    features_noise = generator.add_white_noise(attention_features, noise)
    print("Feature noise: ", features_noise.shape)
    pred_traj, _ = generator.process_decoder(features_noise)

    print("Pred trajectories: ", pred_traj.shape)

    # return pred_traj

## Multi-Layer Perceptron

def test_mlp():
    opt = {
        "dim_list": [2, 64],
        "activation": 'relu',
        "batch_norm": False,
        "dropout": 0
    }
    opt = Prodict.from_dict(opt)
    mlp = MLP(**opt)
    print(mlp)

## Encoder

def test_encoder():
    opt = {
        "num_layers": 1,
        "hidden_dim": 32,
        "emb_dim": 2,
        "dropout": 0.4,
        "mlp_config": {
            "dim_list": [2, 16],
            "activation": 'relu',
            "batch_norm": False,
            "dropout": 0.4
        }
    }
    encoder = Encoder(**opt)
    print(encoder)

## Decoder

def test_decoder():

    input_data = np.random.randn(768,2)
    input_data = torch.from_numpy(input_data).to(device).float()

    config_file.sophie.generator.decoder.linear_3.input_dim = batch_size*2*config_file.sophie.generator.social_attention.linear_decoder.out_features
    config_file.sophie.generator.decoder.linear_3.output_dim = batch_size*na

    print(">>>>>>>>>>>>>>>>>: ", config_file.sophie.generator.decoder)
    sophie_decoder = Decoder(config_file.sophie.generator.decoder).to(device)
    print(">>>>>>>>>>>>>>>>> sophie_decoder: ", sophie_decoder)
    trajectories, state_tuple_1 = sophie_decoder(input_data)
    print("trajectories: ", trajectories.shape)
    #print("state_tuple_1: ", state_tuple_1)

## GAN generator

def test_sophie_generator():

    # Image

    image_test = torch.rand(batch_size, im_channels, im_height, im_width).to(device) # batch, channel, H, W

    # Trajectories

    trajectories_test = 10 * np.random.randn(po, na*batch_size, tfd)
    trajectories_test = torch.from_numpy(trajectories_test).to(device).float()
    # print("Previous trajectories: ", trajectories_test, trajectories_test.shape)

    # config_file.sophie.generator.decoder.linear_3.input_dim = batch_size*2*config_file.sophie.generator.social_attention.linear_decoder.out_features
    # config_file.sophie.generator.decoder.linear_3.output_dim = batch_size*na

    # print("Linear 3 Decoder: ", config_file.sophie.generator.decoder.linear_3)

    past_observations = config_file.hyperparameters.obs_len
    num_agents = config_file.hyperparameters.number_agents
    config_file.sophie.generator.social_attention.linear_decoder.out_features = past_observations * num_agents

    generator = SoPhieGenerator(config_file.sophie.generator)
    generator.set_num_agents(config_file.hyperparameters.number_agents)
    generator.build()
    generator.to(device)
    pred_fake_trajectories = generator.forward(image_test,trajectories_test)

    print("pred_fake_trajectories: ", pred_fake_trajectories, pred_fake_trajectories.shape)

    return pred_fake_trajectories

## GAN discriminator

def test_sophie_discriminator():
    """
    """

    trajectories = 10 * np.random.randn(fo, na*batch_size, tfd) # 8, 32, 2
    trajectories = torch.from_numpy(trajectories).to(device).float()

    discriminator = SoPhieDiscriminator(config_file.sophie.discriminator)
    discriminator.build()
    discriminator.to(device)
    discriminator.forward(trajectories)

def test_aiodrive_dataset():
    data = AioDriveDataset("./data/datasets/aiodrive/aiodrive_Car/train",
        videos_path="./data/datasets/aiodrive/aiodrive_image_front_trainval/image_2")
    print(data)
    batch_size = 8
    loader_num_workers = 0
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate_image_aiodrive)

    print("loader: ", loader)
    print("device: ", device)
    t0 = time.time()
    for batch in loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end, _, frames) = batch

        print("> ", obs_traj.shape, pred_traj_gt.shape, frames.shape, frames[0])
        assert 0==1, "tesu"
        # t1 = time.time()
        # while(t1 - t0 < 120):
        #     print(t1-t0)
        #     t1 = time.time()
        #assert 1 == 0, "aiie"

def test_aiodrive_frames():
    windows_frames = [30,180,330,480,630,780,930]
    # data = AioDriveDataset("./data/datasets/aiodrive/aiodrive_Car/test",windows_frames=windows_frames,phase="testing",split="test")

    data = AioDriveDataset("./data/datasets/aiodrive/aiodrive_Car/train",windows_frames=windows_frames,phase="training",split="train")

def load_id_frame():
    vi_path = "/home/fkite/git-personal/SoPhie/data/datasets/aiodrive/aiodrive_image_front_trainval/image_2"
    extension = "png"
    id_frame = torch.load("id_frame_example.pt")

    def get_folder_name(video_path, seq_name):
        town = str(int(seq_name/1000))
        seq = str(int(seq_name%1000))
        folder = "Town{}_seq{}".format(town.zfill(2), seq.zfill(4))
        full_path = os.path.join(video_path, folder)
        return full_path

    folder_name_ex = get_folder_name(vi_path, 3015)
    #print("folder_name_ex: ",folder_name_ex)
        
    # print("id_frame ", id_frame.shape, id_frame)
    # print("id_frame 0 ", id_frame[:,0,2].shape, id_frame[:,1,1])
    # print("id_frame 0 ", id_frame[0].shape, id_frame[0])
    # print("id_frame 1 ", id_frame[0][0].shape, id_frame[0][0])
    # print("id_frame 2 ", id_frame[0][0][0].shape, id_frame[0][0][0])
    print("url: ", vi_path, extension)
    frames = torch.load("frame_ex.pt")
    print("frames: ", type(frames), frames)
    def load_images(video_path, frames, extension, new_shape=(600,600)):
        frames_list = []
        for frame in frames:
            folder_name = get_folder_name(video_path, frame[0].item())
            image_id = str(int(frame[1].item()))
            image_url = os.path.join(folder_name, "{}.{}".format(image_id.zfill(6), extension))
            frame = cv2.imread(image_url)
            frame = cv2.resize(frame, new_shape)
            frames_list.append(np.expand_dims(frame, axis=0))
        frames_arr = np.concatenate(frames_list, axis=0)
        frames_arr = torch.from_numpy(frames_arr).type(torch.float32).permute(0, 3, 1, 2)
        return frames_list

    frames_list = load_images(vi_path, list(frames), extension)
    # print("obs_traj: ", obs_traj.shape, obs_traj) # 8, 182, 2

class AutoTree(dict):
    """
    Dictionary with unlimited levels
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def condition(x,class_name):
    return x==class_name

def test_json():
    prediction_length = '10' # [10,20,50]
    classes = {"Car":0, "Cyc":1, "Ped":2, "Mot":3, "Dum":4} # Car, Ped, Mot, Cyc, Dummy
    seq_name = 'Town07_seq0000' #['Town07_seq0000','Town07_seq0001','Town07_seq0002']
    seq_frame = '50' # [50,200,350,500]
    trajectory_sample = '0'
    prob = 0.5

    real_objects_class = np.round(np.random.uniform(0,3,27)).reshape(1,-1)
    dummy_objects_class = np.ones((1,5))*4 
    object_class = np.hstack([real_objects_class,dummy_objects_class]).astype(int)
    print("Object class: ", object_class, object_class.shape)

    real_objects_id = np.random.randint(0,100,27).reshape(1,-1)
    dummy_objects_id = np.ones((1,5))*-1
    object_id = (np.hstack([real_objects_id,dummy_objects_id])).astype(int)
    print("Object id: ", object_id, object_id.shape)

    pred_fake_trajectories = test_sophie_generator()
    print("Shape: ", pred_fake_trajectories.shape)

    object_indexes = []
    data = AutoTree()

    # Lv 0: Prediction length (10, 20, 50)
    # Lv 1: Object class: Car, Ped, Cyc, Mot
    # Lv 2: Sequence name: Town07_seq0000, etc.
    # Lv 3: First frame of the current prediction window (if using 0 to 49 to predict, it will be 50)
    # Lv 4: Trajectory sample for each object (Possible trajectories)
    # Lv 5: Object ID
    # Lv 6: State (N (N = 10 frames in the future) x 2 (x,y BEV positions)), 
    #       Prob: Probability value for this particular trajectory sample

    # {pred_len1: {obj_class: {seqname: {frame: {sample: {ID: {'state': N x 2, 'prob': 0.83}}}}}}}

    print("\n")

    for key,value in classes.items():
        indexes = np.where(np.array([condition(xi,value) for xi in object_class]))[1]

        agent_dict = {}
        # print("Indexes: ", indexes)
        for i in range(pred_fake_trajectories.shape[1]): 
            if i in indexes:
                ground_pos = []
                for j in range(pred_fake_trajectories.shape[0]): 
                    ground_pos.append(pred_fake_trajectories[j,i,:].tolist()) # x,y
                # We assume here i is the identifier
                # print("Ground pos: ", ground_pos)
                aux_dict = {}
                aux_dict['state'] = ground_pos
                aux_dict['prob'] = prob
                agent_dict[str(object_id[0,i])] = aux_dict

                data[prediction_length][key][seq_name][seq_frame][trajectory_sample] = agent_dict

    print("\n")
    print("Data: ", data)

    with open('data.json', 'w') as fp:
        json.dump(data, fp)

def test_evaluate_json_aiodrive():
    prediction_length = '10'
    trajectory_sample = '0'
    data = torch.load("data_sophie_trainval.pt")
    classes = {"Car":0, "Cyc":1, "Mot":2, "Ped":3, "Dum":4} # Car, Ped, Mot, Cyc, Dummy
    prob = 1.0
    batch_size_aiodrive = 8

    generator = SoPhieGenerator(config_file.sophie.generator)
    generator.build()
    generator.to(device)

    obs_traj = data['obs_traj']
    frames = data['frames']

    predicted_trajectories = generator.forward(frames,obs_traj) # 12 x batch_size*na (8x32) x 2

    json_dict = AutoTree()

    for element in range(batch_size_aiodrive):
        # obs_traj = data['obs_traj'][:,na*element:na*(element+1),:] # 8 x batch_size*na (8x32) x 2 -> 8 x 32 x 2
        # frames = data['frames'][element].reshape(1,3,600,600) # batch_size (8) x 3 x 600 x 600 -> 1 x 3 x 600 x 600
        object_cls = data['object_cls'][element].reshape(1,-1).cpu().data.numpy() # 8 x 32 -> 1 x 32
        seq = data['seq'][element].reshape(1,-1).cpu().data.numpy() # 8 x 2 -> 1 x 2
        obj_id = data['obj_id'][element].reshape(1,-1).cpu().data.numpy() # 8 x 32 -> 1 x 32 
        pred_fake_trajectories = predicted_trajectories[:,na*element:na*(element+1),:]
        # pred_fake_trajectories = generator.forward(frames,obs_traj) # 12 x 32 x 2

        object_indexes = []

        xx = str(int(seq[0,0]/1000))
        yyyy = str(int(seq[0,0]%1000))

        seq_name = ""

        if len(str(seq[0,0])) == 5:
            seq_name = "Town"+xx.zfill(2)+"HD"+"_seq"+yyyy.zfill(4)
        else:
            seq_name = "Town"+xx.zfill(2)+"_seq"+yyyy.zfill(4)

        seq_frame = str(int(seq[0,1]))

        for key,value in classes.items():
            indexes = np.where(np.array([condition(xi,value) for xi in object_cls]))[1]

            agent_dict = {}
            # print("Indexes: ", indexes)
            for i in range(pred_fake_trajectories.shape[1]): 
                if i in indexes:
                    ground_pos = []
                    for j in range(pred_fake_trajectories.shape[0]): 
                        ground_pos.append(pred_fake_trajectories[j,i,:].tolist()) # x,y
                    # We assume here i is the identifier
                    # print("Ground pos: ", ground_pos)
                    aux_dict = {}
                    aux_dict['state'] = ground_pos
                    aux_dict['prob'] = prob
                    agent_dict[str(int(obj_id[0,i]))] = aux_dict

                    json_dict[prediction_length][key][seq_name][seq_frame][trajectory_sample] = agent_dict

    with open('test_results.json', 'w') as fp:
        json.dump(json_dict, fp)

def load_id_frame_ex():
    id_frames = torch.load("id_frame_example.pt") # original: 32, 3, 20 -> dataloader
    print("id_frames: ", id_frames.shape, id_frames)

def evaluate_json():
    with open("results/aiodrive/submission_95_percent_training_all_objects_1s_2s.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    jsonObject.pop('10')

    with open('results/aiodrive/submission_95_percent_training_all_objects_2s.json', 'w') as fp:
        json.dump(jsonObject, fp)

    # pl = ['10','20','50']
    # cl = ['Car','Cyc','Ped','Mot']

    # for p in pl:
    #     for c in cl:
    #         for key in jsonObject[p][c].keys():
    #             num_keys = jsonObject[p][c][key].keys()
    #             for key_a in num_keys:
    #                 ts = list(jsonObject[p][c][key][key_a].keys())
                    
    #                 if '0' in ts and '1' in ts:
    #                     continue
    #                 else:
    #                     print("ts: ", ts)
    #                     print("Wrong")

def test_autotree():
    json_dict = AutoTree()

    json_dict['apple']['tree']['red'] = 5
    json_dict['orange']['new_tree']['orange_colour'] = 7
    print("json dict: ", json_dict)
    print("keys: ", json_dict.keys())

def load_npy():
    # MIA_10316_driveable_area_mat_2019_05_28
    # MIA_10316_ground_height_mat_2019_05_28
    # MIA_10316_halluc_bbox_table
    # MIA_10316_npyimage_to_city_se2_2019_05_28

    # PIT_10314_driveable_area_mat_2019_05_28
    # PIT_10314_ground_height_mat_2019_05_28
    # PIT_10314_halluc_bbox_table
    # PIT_10314_npyimage_to_city_se2_2019_05_28

    img_array = np.load("./data/datasets/argoverse/hd-maps/map_files/PIT_10314_halluc_bbox_table.npy") # .astype(np.uint8)
    print("Shape: ", img_array.shape)
    print("Image: PIT_10314_halluc_bbox_table")
    imageio.imsave("/home/robesafe/shared_home/PIT_10314_halluc_bbox_table.png",img_array)

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

def test_argoverse_csv():
    # Create a dict and store in JSON format to get the conversion between our object_id and Argoverse track_id
    track_id_coded_flag = True
    # Data -> Numpy array (seq x timestamp x (frame_id | object_id | x | y))
    data_structure_flag = False
    debug_time = False
    filter_by_distance = False
    load_checkpoint = False

    split = "val" # train, val, test_obs
    folder = "data/datasets/argoverse/motion-forecasting/" + split + "/data/"
    coding_folder = "data/datasets/argoverse/motion-forecasting/" + split + "/track_id_coded/"

    parent_folder = '/'.join(os.path.normpath(folder).split('/')[:-1])

    files, num_files = load_list_from_folder(folder)
    file_id_list = []
    for file_name in files:
        file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
        file_id_list.append(file_id)
    print("Num files: ", num_files)

    file_id_list.sort()

    if track_id_coded_flag:
        limit = 10000
        track_id_coded = dict()
        seq_start = 0
        seq_end = 0
        new_json = True

        for i,file_id in enumerate(file_id_list):
            if new_json:
                track_id_coded = dict()
                start = time.time()
                seq_start = file_id
                new_json = False
            track_file = folder + str(file_id) + ".csv"
            df = pd.read_csv(track_file)
            col_track_id = np.array(df['TRACK_ID'])
            object_type = np.array(df['OBJECT_TYPE'])

            # Get AV and AGENT ids in Argoverse format

            agents = dict()
            cont = 0

            first_av_row = np.where(object_type == 'AV')[0][0]
            agents[cont] = col_track_id[first_av_row] # 0 == AV
            cont += 1
            first_agent_row = np.where(object_type == 'AGENT')[0][0]
            agents[cont] = col_track_id[first_agent_row] # 1 == AGENT
            cont += 1

            for row in col_track_id:       
                if row not in agents.values(): # From 2 to n, OTHERS
                    agents[cont] = row
                    cont += 1 

            track_id_coded[track_file] = agents
            
            # Store JSON

            if (i % limit == 0 and i > 0) or i == num_files-1:
                seq_end = file_id
                track_id_coded['sequences'] = "/seq_" + str(seq_start) + "_" + str(seq_end)
                
                print(f"Store JSON from sequence {seq_start} to {seq_end}")
                print("JSON length: ", len(track_id_coded)-1) # -1 since one key represents the stored sequences

                if not os.path.isdir(parent_folder + "/track_id_coded"):
                    os.mkdir(parent_folder + "/track_id_coded")
                track_id_coded_json = parent_folder + "/track_id_coded" + "/seq_" + str(seq_start) + "_" + str(seq_end) + ".json"
                print(track_id_coded_json)
                with open(track_id_coded_json, "w") as outfile:
                    json.dump(track_id_coded, outfile)
                new_json = True

                end = time.time()
                print(f"Time consumed: {end-start}\n")

    if data_structure_flag:
        dist_threshold = 20
        checkpoint = 10000

        track_id_coded_files, num_track_id_coded_files = load_list_from_folder(coding_folder)

        # Load JSON files

        json_files = []
        for track_id_coded_file in track_id_coded_files:
            with open(track_id_coded_file) as json_file:
                data = json.load(json_file)
                json_files.append(data)

        seqs = np.array([]).reshape(2,0)
        for json_file in json_files:
            # seq_name = os.path.normpath(file_name).split('/')[-1].split('.')[0]
            seq_name = json_file['sequences']
            start = int(os.path.normpath(seq_name).split('_')[1])
            end = int(os.path.normpath(seq_name).split('_')[-1].split('.')[0])
            seq = np.array([start,end]).reshape(-1,1)
            seqs = np.hstack([seqs,seq])

        print("JSON files loaded")

        # Load checkpoint

        last_file_id = -1

        if load_checkpoint:
            try:
                obs_trajectories_file = "data/datasets/argoverse/motion-forecasting/train/obs_trajectories.npy"
                # obs_trajectories_file = "shared_home/benchmarks/argoverse/motion-forecasting/train/obs_trajectories.npy"
                with open(obs_trajectories_file, 'rb') as obs_file:
                    obs_trajectories = np.load(obs_file)
                    for i in range(obs_trajectories.shape[0]-1,0,-1):
                        if obs_trajectories[i,1] == np.float64(-1):
                            last_file_id = int(obs_trajectories[i,0])
                            csv_start = last_file_id + 1
                            print(f"Start from {last_file_id}.csv")
                            break 
            except:
                obs_trajectories = np.array([]).reshape(0,4)
                last_file_id = -1
                csv_start = 1
                print(f"Start from 1.csv")

        # Calculate the trajectories

        print_time = False
        npy_store_start = time.time()
        seq_separators_only_indexes = [] # The trajectories file DO NOT include separators
        previous_index = 0
        seq_separators_only_indexes.append(previous_index)
        aux_traj = []
        start_npy = True
        
        for t,file_id in enumerate(file_id_list):
            if file_id > last_file_id:
                if start_npy:
                    csv_start = file_id
                    start_npy = False
                if debug_time: print("--------------------")
                if debug_time: print("file id: ", file_id)
                a_start = time.time()
                track_file = folder + str(file_id) + ".csv"
                # print(f"Analyzing {file_id}.csv")
                data = dict()
                for k in range(seqs.shape[1]):
                    if file_id >= seqs[0,k] and file_id <= seqs[1,k]:
                        # aux_json_file = coding_folder + 'seq_' + str(int(seqs[0,k])) + '_' + str(int(seqs[1,k])) + '.json'
                        # with open(aux_json_file) as json_file:
                        #     data = json.load(json_file)
                        data = json_files[k]
                a_end = time.time()
                if debug_time: print(f"Time consumed from first part: {a_end-a_start}")
                a_start = time.time()
                df = pd.read_csv(track_file)


                folder = "data/datasets/argoverse/motion-forecasting/train/data/"
                # folder = "shared_home/benchmarks/argoverse/motion-forecasting/train/data/"
                track_file = folder + str(file_id) + ".csv"
                
                dict1 = data[track_file] # Standard IDs (Keys) to Argoverse values (Values)
                dict2 = dict(zip(dict1.values(), dict1.keys())) # Argoverse values (Keys) to standard IDs (Values)

                timestamps = np.array(df['TIMESTAMP']).reshape(-1,1)
                track_ids = np.array(df['TRACK_ID']).reshape(-1,1)
                x_pos = np.array(df['X']).reshape(-1,1)
                y_pos = np.array(df['Y']).reshape(-1,1)

                # Additional column to represent the class: 0 == AV, 1 == AGENT, 2 == OTHERS
                
                object_type = np.array(df['OBJECT_TYPE']).reshape(-1,1)
                object_class = [0 if obj=="AV" else 1 if obj=="AGENT" else 2 for obj in object_type]
                object_class = np.array(object_class).reshape(-1,1)

                coded_track_ids = np.vectorize(dict2.get)(track_ids).astype(np.int64)

                print("Coded: ", coded_track_ids)
                assert 1 == 0
                
                seq_sophie_format = (timestamps,coded_track_ids,x_pos,y_pos,object_class)
                seq_sophie_format = np.concatenate(seq_sophie_format, axis=1)
                # print("seq_sophie_format: ", seq_sophie_format, seq_sophie_format.shape)
                a_end = time.time()
                if debug_time: print(f"Time consumed from second part: {a_end-a_start}")

                a_start = time.time()

                # sequence_separator = np.array([file_id,-1,-1,-1]).reshape(1,4)
                # seq_sophie_format = np.concatenate([sequence_separator,seq_sophie_format]) 
                aux_traj.append(seq_sophie_format)

                # Get separators

                if t != len(file_id_list) - 1: # Save all except the last one, since it represents the end
                    seq_len = timestamps.shape[0]
                    previous_index += seq_len
                    seq_separators_only_indexes.append(previous_index)

                a_end = time.time()
                if debug_time: print(f"Time consumed from third part: {a_end-a_start}")

                if (t > 0 and t % checkpoint == 0) or(t == len(file_id_list) - 1):
                    a_start = time.time()
                    aux_traj = np.concatenate(aux_traj, axis=0)
                    # obs_trajectories = np.concatenate([obs_trajectories, aux_traj])
                    a_end = time.time()

                    npy_store_end = time.time()
                    csv_end = file_id
                    print(f"Time consumed from {csv_start}.csv to {csv_end}.csv: {npy_store_end-npy_store_start}")
                    npy_store_start = npy_store_end
                    start_npy = True

                    # Save checkpoint

                    if not os.path.isdir(parent_folder + "/obs_trajectories"):
                        os.mkdir(parent_folder + "/obs_trajectories")

                    obs_trajectories_file_root = "data/datasets/argoverse/motion-forecasting/train/obs_trajectories/obs_trajectories_"
                    # obs_trajectories_file_root = "data/datasets/argoverse/motion-forecasting/train/obs_trajectories/obs_trajectories_"
                    # obs_trajectories_file_root = "shared_home/benchmarks/argoverse/motion-forecasting/train/obs_trajectories/obs_trajectories_"
                    obs_trajectories_file = obs_trajectories_file_root + str(csv_start) + "_" + str(csv_end) + "_csv" + ".npy"
                    with open(obs_trajectories_file, 'wb') as obs_file:
                        # np.save(obs_file, obs_trajectories)
                        np.save(obs_file, aux_traj)

                    aux_traj = []

                    # if t == len(file_id_list) - 1:
                    #     seq_separators_file = "data/datasets/argoverse/motion-forecasting/train/sequence_separators.npy"
                    #     seq_separators_only_indexes = np.array(seq_separators_only_indexes)
                    #     with open(seq_separators_file, 'wb') as seq_file:
                    #         np.save(seq_file, seq_separators_only_indexes)
        # print("obs_trajectories: ", obs_trajectories, obs_trajectories.shape)
        # obs_trajectories_file = "shared_home/benchmarks/motion-forecasting/train/obs_trajectories.npy"
        # with open(obs_trajectories_file, 'wb') as obs_file:
        #     np.save(obs_file, obs_trajectories)

def concat_npy_files():
    # folder = "data/datasets/argoverse/motion-forecasting/train/obs_trajectories_without_separators"
    folder = "data/datasets/argoverse/motion-forecasting/train/obs_trajectories"
    # folder = "shared_home/benchmarks/argoverse/motion-forecasting/train/obs_trajectories"
    files, num_files = load_list_from_folder(folder)

    starts = []
    for file_name in files:
        seq_name = os.path.normpath(file_name).split('/')[-1].split('.')[0]
        start = int(os.path.normpath(seq_name).split('_')[2])
        starts.append(start)

    starts = np.array(starts)
    sort_index = np.argsort(starts)

    obs_trajectories_list = []
    rows = 0
    for t in sort_index: # Concat sorted .npy files
        obs_trajectories_file = files[t]
        print("File: ", files[t])
        with open(obs_trajectories_file, 'rb') as obs_file:
            obs_trajectories = np.load(obs_file)
            print("Shape: ", obs_trajectories.shape)
            rows += obs_trajectories.shape[0]
            obs_trajectories_list.append(obs_trajectories)

    print("\nNumber of rows: ", rows)
    joined_obs_trajectories = np.concatenate(obs_trajectories_list, axis=0)

    obs_trajectories_file = "data/datasets/argoverse/motion-forecasting/train/joined_obs_trajectories.npy"
    # obs_trajectories_file = "shared_home/benchmarks/argoverse/motion-forecasting/train/joined_obs_trajectories.npy"
    with open(obs_trajectories_file, 'wb') as obs_file:
        np.save(obs_file, joined_obs_trajectories)

def store_city():
    """
    """

    folder = "data/datasets/argoverse/motion-forecasting/train/data/"
    files, num_files = load_list_from_folder(folder)
    file_id_list = []
    for file_name in files:
        file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
        file_id_list.append(file_id)
    print("Num files: ", num_files)

    file_id_list.sort()

    # city_id_list = []
    city_id_array = np.zeros((len(file_id_list)))

    start = time.time()

    for i,file_id in enumerate(file_id_list):
        track_file = folder + str(file_id) + ".csv"
        df = pd.read_csv(track_file)

        city = np.array(df['CITY_NAME']).reshape(-1,1)
        city = city[0,0]

        print("city: ", city)

        if city == "PIT":
            city_id_array[i] = 0
        else:
            city_id_array[i] = 1

        print("i: ", i)

    city_id_file = "data/datasets/argoverse/motion-forecasting/train/city_id.npy"
    with open(city_id_file, 'wb') as city_id_file:
        np.save(city_id_file, city_id_array)

    end = time.time()

    print(f"Time consumed: {end-start}")

def separate_in_sequences(joined_obs_trajectories):
    if np.int64(joined_obs_trajectories[1]) == -1: return 1
    else: return 0

def separate_in_sequences_v2(obj_id):
    if np.int64(obj_id) == -1: return 1
    else: return 0

def distance_filter(sequences, relative_batch_separators, dist_threshold):

    print("-----------------> Distance filter")
    filtered_sequences = np.array([]).reshape(0,4)
    new_seq_separators = []
    new_seq_separators.append(0)
    aux_seq_separator = 0

    print("Shape seq: ", sequences.shape)
    print("relative batch separators: ", relative_batch_separators)

    for i, batch_separator in enumerate(relative_batch_separators):
        print("batch separator: ", batch_separator)
        if i < len(relative_batch_separators)-1:
            sequence = sequences[relative_batch_separators[i]:relative_batch_separators[i+1],:]
        else:
            sequence = sequences[relative_batch_separators[i]:,:]

        print("Seq: ", sequence[0,:], sequence.shape)
        to_filter_by_distance = np.zeros((sequence.shape[0]))
        ref_x, ref_y = 0, 0
        for t in range(sequence.shape[0]):
            if sequence[t,1] == 0: # The AV starts the observations of a sequence and also for each timestamp
                av_x = sequence[t,2]
                av_y = sequence[t,3]
                # print("--------------")
                # print("ref: ", av_x,av_y)
            else:
                obj_x = sequence[t,2]
                obj_y = sequence[t,3]

                dist = math.sqrt(pow(obj_x-av_x,2)+pow(obj_y-av_y,2))
                # print("Dist to AV: ", dist)
                if dist > dist_threshold:
                    to_filter_by_distance[t] = 1

        # print("Before filtering: ", sequence.shape)
        filtered_sequence = sequence[to_filter_by_distance[:] != 1]
        filtered_sequences = np.concatenate([filtered_sequences, filtered_sequence])
        near_objects = np.count_nonzero(to_filter_by_distance == 0)
        print("Objects after dist filter (including ego obs): ", filtered_sequence.shape[0])

        if i < len(relative_batch_separators)-1:
            new_seq_separator = filtered_sequence.shape[0]
            aux_seq_separator += new_seq_separator
            new_seq_separators.append(aux_seq_separator)
        # print("After filtering: ", sequence.shape)
    return filtered_sequences, new_seq_separators

def dummies_filter(filtered_by_distance_sequences, new_seq_separators, agents_per_obs=9):
    """
    """

    print("-----------------> Dummy filter")

    print(filtered_by_distance_sequences.shape)

    seq_separator_pre = 0

    dummy_filtered_sequences = np.array([]).reshape(0,4)

    # print("New seq separators: ", new_seq_separators)
    for t, seq_separator in enumerate(new_seq_separators):
        if t < len(new_seq_separators) - 1:
            # print("seq separators: ", new_seq_separators[t], new_seq_separators[t+1])
            filtered_by_distance_sequence = filtered_by_distance_sequences[new_seq_separators[t]:new_seq_separators[t+1],:]
        else:
            # print("from seq_separator to the end: ", new_seq_separators[t])
            filtered_by_distance_sequence = filtered_by_distance_sequences[new_seq_separators[t]:,:]

        # print("filtered by distance seq: ", filtered_by_distance_sequence.shape)#, filtered_by_distance_sequence)
        # dummy_filtered_sequence = 

        obs_windows = np.where(filtered_by_distance_sequence[:,1] == 0)[0]

        # print("obs windows: ", obs_windows, len(obs_windows))

        for i in range(len(obs_windows)):
            # print("i: ", i)
            if i < len(obs_windows) - 1:
                agents_in_obs = obs_windows[i+1] - obs_windows[i] - 1
                sub_sequence = filtered_by_distance_sequence[obs_windows[i]:obs_windows[i+1],:] # Including ego-vehicle
            else:
                agents_in_obs = filtered_by_distance_sequence.shape[0] - obs_windows[i] - 1
                sub_sequence = filtered_by_distance_sequence[obs_windows[i]:,:] # Including ego-vehicle
            # print("agents in obs: ", agents_in_obs)

            if agents_in_obs < agents_per_obs: # We introduce dummy data
                timestamp = sub_sequence[0,0]
                dummy_agents = agents_per_obs - agents_in_obs
                dummy_array = np.array([timestamp,-1,-1.0,-1.0])
                dummy_array = np.tile(dummy_array,dummy_agents).reshape(-1,4)
                dummy_sub_sequence = np.concatenate([sub_sequence,dummy_array])
                # print("Pre: ", sub_sequence.shape)
                # print("Post: ", dummy_sub_sequence.shape)
            elif agents_in_obs == agents_per_obs:
                dummy_sub_sequence = sub_sequence
            else:
                # Sort agents by distance
                # print("\n..................")
                agents_dist = []

                for t in range(sub_sequence.shape[0]):
                    if sub_sequence[t,1] == 0: # The AV starts the observations of a sequence and also for each timestamp
                        av_x = sub_sequence[t,2]
                        av_y = sub_sequence[t,3]
                        # print("--------------")
                        # print("ref: ", av_x,av_y)
                    else:
                        obj_x = sub_sequence[t,2]
                        obj_y = sub_sequence[t,3]

                        dist = math.sqrt(pow(obj_x-av_x,2)+pow(obj_y-av_y,2))
                        agents_dist.append(dist)
                agents_dist = np.array(agents_dist)
                sorted_indeces = np.argsort(agents_dist)
                # print("Agents dist: ", agents_dist)
                # print("Sorted: ", sorted_indeces)

                to_delete_indeces = sorted_indeces[agents_per_obs:] # Only keep the closest agents_per_obs agents

                # print("to delete indeces: ", to_delete_indeces)
                # print("sub sequence shape: ", sub_sequence.shape)
                dummy_sub_sequence = np.delete(sub_sequence,to_delete_indeces,axis=0)
                # print("dummy sequence: ", dummy_sub_sequence)
                # print("shape after cropping: ", dummy_sub_sequence.shape)
            
            dummy_filtered_sequences = np.concatenate([dummy_filtered_sequences,dummy_sub_sequence])
            # print("Shape pre: ", dummy_filtered_sequences.shape)

    return dummy_filtered_sequences  

def relative_displacements(batch_size, fixed_sized_sequences, num_last_obs=19):
    """
    """

    print("-----------------> Relative displacements")

    num_agents_per_obs = 10 # including ego-vehicle
    num_obs = 50
    num_positions = num_obs * num_agents_per_obs

    relative_sequences = np.array([]).reshape(0,4)
    ego_vehicle_origin = np.array([]).reshape(0,2)

    for i in range(batch_size):
        if i < batch_size - 1:
            sequence = fixed_sized_sequences[num_positions*i:num_positions*(i+1),:]
        else:
            sequence = fixed_sized_sequences[num_positions*i:,:]

        origin_x = sequence[num_last_obs*num_agents_per_obs,2]
        origin_y = sequence[num_last_obs*num_agents_per_obs,3]
        origin = np.array([origin_x, origin_y]).reshape(1,2)
        ego_vehicle_origin = np.concatenate([ego_vehicle_origin, origin])

        for j in range(sequence.shape[0]):
            if np.int64(sequence[j,1]) != -1:
                sequence[j,2] -= origin_x
                sequence[j,3] -= origin_y
        relative_sequences = np.concatenate([relative_sequences,sequence])
    return relative_sequences, ego_vehicle_origin

def read_joined_obs_trajectories():
    get_seq_separators = False
    obs_trajectories_file = "data/datasets/argoverse/motion-forecasting/train/joined_obs_trajectories.npy"
    # seq_separators_file = "data/datasets/argoverse/motion-forecasting/train/seq_separators_only_indexes.npy"
    seq_separators_file = "data/datasets/argoverse/motion-forecasting/train/sequence_separators.npy"
    # obs_trajectories_file = "shared_home/benchmarks/argoverse/motion-forecasting/train/joined_obs_trajectories.npy"
    # seq_separators_file = "shared_home/benchmarks/argoverse/motion-forecasting/train/seq_separators_only_indexes.npy"
    with open(obs_trajectories_file, 'rb') as obs_file:
       joined_obs_trajectories =  np.load(obs_file)
       print("joined_obs_trajectories Shape: ", joined_obs_trajectories.shape)

    dist_threshold = 50

    if get_seq_separators:
        # A

        # a_start = time.time()
        # # seq_separators = np.zeros((joined_obs_trajectories.shape[0]))
        # seq_separators = []
        # for i,traj in enumerate(joined_obs_trajectories):
        #     # if np.int64(traj[1]) == -1: seq_separators[i] = 1 
        #     if np.int64(traj[1]) == -1: seq_separators.append(i)
        # a_end = time.time()
        # print(f"Time consumed by first approach: {a_end-a_start}")
        # print(f"Time consumed by first approach (per iteration): {(a_end-a_start)/joined_obs_trajectories.shape[0]}\n")
        # print("seq_separators: ", seq_separators[:10]) 

        # # # B

        # b_start = time.time()
        # seq_separators = np.apply_along_axis(separate_in_sequences, 1, joined_obs_trajectories)
        # b_end = time.time()
        # print(f"Time consumed by second approach: {b_end-b_start}")
        # print(f"Time consumed by second approach (per iteration): {(b_end-b_start)/joined_obs_trajectories.shape[0]}\n")
        # print("seq_separators: ", seq_separators[:10]) 

        # # # C

        # c_start = time.time()
        # ids = joined_obs_trajectories[:,1]
        # vectorized_separate_sequences = np.vectorize(separate_in_sequences_v2)
        # seq_separators = vectorized_separate_sequences(ids)
        # c_end = time.time()
        # print(f"Time consumed by third approach: {c_end-c_start}")
        # print(f"Time consumed by third approach (per iteration): {(c_end-c_start)/joined_obs_trajectories.shape[0]}\n")
        # print("seq_separators: ", seq_separators[:10]) 

        # # D 

        d_start = time.time()
        ids = joined_obs_trajectories[:,1]
        # seq_separators = np.zeros((joined_obs_trajectories.shape[0]))
        seq_separators = []
        for i,obj_id in enumerate(ids):
            # if np.int64(obj_id) == -1: seq_separators[i] = 1
            if np.int64(traj[1]) == -1: seq_separators.append(i)
        d_end = time.time()
        print(f"Time consumed by fourth approach: {d_end-d_start}")
        print(f"Time consumed by fourth approach (per iteration): {(d_end-d_start)/joined_obs_trajectories.shape[0]}\n")

        # E (Very efficient)

        e_start = time.time()
        ids = joined_obs_trajectories[:,1]
        seq_separators = np.where(ids == -1)
        # print("Shape: ", len(seq_separators), tuple(seq_separators))
        # print("seq separators: ", seq_separators[:200])
        seq_separators = np.array(seq_separators)
        e_end = time.time()
        print(f"Time consumed by fifth approach: {e_end-e_start}")
        print(f"Time consumed by fifth approach (per iteration): {(e_end-e_start)/joined_obs_trajectories.shape[0]}")

        print("seq_separators: ", seq_separators.shape)
        with open(seq_separators_file, 'wb') as seq_file:
            np.save(seq_file, seq_separators)
    else:
        with open(seq_separators_file, 'rb') as seq_file:
            seq_separators = np.load(seq_separators_file).reshape(-1)
        batch_size = 1
        print("seq separators Shape: ", seq_separators.shape)
        last_end = 0

        print("seq separators: ", seq_separators[:50])

        for cont in range(int(len(seq_separators)/batch_size+1)): # TODO: Check this
            
            start = seq_separators[cont*batch_size]
            end = seq_separators[(cont+1)*batch_size]
            last_start = start
            batch_separators = seq_separators[cont*batch_size:(cont+1)*batch_size]

            sequences = joined_obs_trajectories[start:end,:]

            # print("cont: ", cont) 
            # print("start, end: ", start, end)
            # print("sequences: ", sequences.shape)
            # print("batch separators: ", batch_separators)
            relative_batch_separators = batch_separators - last_start
            # print("relative batch separators: ", relative_batch_separators)
            # print("last end: ", last_start)
            # print("\n")

            filtered_by_distance_sequences, new_seq_separators = distance_filter(sequences, relative_batch_separators, dist_threshold)
            fixed_sized_sequences = dummies_filter(filtered_by_distance_sequences, new_seq_separators, agents_per_obs=9)
            relative_sequences, ego_vehicle_origin = relative_displacements(batch_size, fixed_sized_sequences, num_last_obs=19)

            print("Relative sequences: ", relative_sequences[:191,:], relative_sequences.shape)
            print("Ego vehicle origin: ", ego_vehicle_origin, ego_vehicle_origin.shape)

            relative_sequences_file = "data/datasets/argoverse/motion-forecasting/train/prueba_miguel/relative_sequences.npy"
            with open(relative_sequences_file, 'wb') as file:
                np.save(relative_sequences_file, relative_sequences)

            ego_origin_file = "data/datasets/argoverse/motion-forecasting/train/prueba_miguel/ego_origin.npy"
            with open(ego_origin_file, 'wb') as file:
                np.save(ego_origin_file, ego_vehicle_origin)



            assert 1 == 0

def mod_seq_separators():
    """
    """
    seq_separators_file = "data/datasets/argoverse/motion-forecasting/train/sequence_separators.npy"
    with open(seq_separators_file, 'rb') as seq_file:
        seq_separators = np.load(seq_separators_file).reshape(-1)

    # seq_separators = np.concatenate([np.zeros((1)),seq_separators]).astype(np.int64)
    seq_separators = seq_separators[:-1]

    # with open(seq_separators_file, 'wb') as seq_file:
    #     np.save(seq_file, seq_separators)


def check_npy():
    with open("data/datasets/argoverse/motion-forecasting/train/after_dataset_processing/object_id_list.npy", 'rb') as aux_file:
    # with open("data/datasets/argoverse/motion-forecasting/train/after_dataset_processing/loss_mask_list.npy", 'rb') as aux_file:
        object_id_list = np.load(aux_file, allow_pickle=True)

    print("object_id_list: ", object_id_list, object_id_list.shape)

def load_csv_number():
    folder = "data/datasets/argoverse/motion-forecasting/train/data/"

    files, num_files = load_list_from_folder(folder)
    file_id_list = []
    for file_name in files:
        file_id = int(os.path.normpath(file_name).split('/')[-1].split('.')[0])
        file_id_list.append(file_id)
    file_id_list.sort()

    index_number = 198435
    csv_number = file_id_list[index_number]
    print("csv number: ", csv_number)

if __name__ == "__main__":
    # test_read_file()
    # test_dataLoader()
    # test_dataLoader_img()
    # test_visual_extractor() 
    # test_joint_extractor() 
    # test_physical_attention_model()
    # test_social_attention_model()
    # test_concat_features() 
    # test_mlp()
    # test_encoder()
    # test_decoder()
    # test_sophie_generator()
    test_sophie_discriminator()
    # test_aiodrive_dataset()
    # test_aiodrive_frames()
    # load_id_frame()
    # test_aiodrive_json()
    # test_json()
    # test_evaluate_json_aiodrive()
    # load_id_frame_ex()
    # evaluate_json()
    # test_autotree()
    # load_npy()
    # test_argoverse_csv()
    # concat_npy_files()
    # read_joined_obs_trajectories()
    # mod_seq_separators()
    # store_city()
    # check_npy()
    # load_csv_number()

    # path_video = "./data/datasets/videos/seq_eth.avi"
    # image_list = read_video(path_video, (600,600))

    # print("image_list: ", type(image_list), len(image_list))
