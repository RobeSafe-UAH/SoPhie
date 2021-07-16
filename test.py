import os
import numpy as np
import yaml
import cv2
import time
import json

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

import torch
from torch import nn
from torch import rand
from torch.utils.data import DataLoader

# Global variables

## CUDA device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Configuration file

with open(r'./configs/sophie_aiodrive.yml') as config_file:
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

po = 8 # Past Observations
fo = 12 # Future Observations
na = 32 # Number of Agents 
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

    # config_file.sophie.generator.decoder.linear_3.input_dim = batch_size*2*config_file.sophie.generator.social_attention.linear_decoder.out_features
    # config_file.sophie.generator.decoder.linear_3.output_dim = batch_size*na

    # print("Linear 3 Decoder: ", config_file.sophie.generator.decoder.linear_3)

    generator = SoPhieGenerator(config_file.sophie.generator)
    generator.build()
    generator.to(device)
    pred_fake_trajectories = generator.forward(image_test,trajectories_test)

    return pred_fake_trajectories

## GAN discriminator

def test_sophie_discriminator():
    """
    """

    trajectories = 10 * np.random.randn(po+fo, na*dlb, tfd) # 8, 32, 2
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

<<<<<<< HEAD
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
=======
def load_id_frame_ex():
    id_frames = torch.load("id_frame_example.pt") # original: 32, 3, 20 -> dataloader
    print("id_frames: ", id_frames.shape, id_frames)

>>>>>>> feature/trainer

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
<<<<<<< HEAD
    # test_sophie_generator()
    # test_sophie_discriminator()
    # test_aiodrive_dataset()
    test_aiodrive_frames()
    # load_id_frame()
    # test_aiodrive_json()
    # test_json()
=======
    #test_sophie_generator()
    #test_sophie_discriminator()
    test_aiodrive_dataset()
    #load_id_frame_ex()
    #load_id_frame()
>>>>>>> feature/trainer

    # path_video = "./data/datasets/videos/seq_eth.avi"
    # image_list = read_video(path_video, (600,600))

    # print("image_list: ", type(image_list), len(image_list))
