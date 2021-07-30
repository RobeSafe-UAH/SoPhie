import argparse
import os
import yaml
import torch
import json
import numpy as np
from pathlib import Path
from prodict import Prodict
from torch.utils.data import DataLoader

from attrdict import AttrDict

from sophie.data_loader.aiodrive.dataset import read_file, seq_collate_image_aiodrive, AioDriveDataset
from sophie.models import SoPhieGenerator
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.utils import relative_to_abs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_path', default='data/datasets/aiodrive/aiodrive_Car/', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--results_path', default='results/aiodrive', type=str)
parser.add_argument('--results_file', default='test_json', type=str)
parser.add_argument('--skip', default=1, type=int)

classes = {"Car":0, "Cyc":1, "Mot":2, "Ped":3, "Dum":-1} # Car, Ped, Mot, Cyc, Dummy

class AutoTree(dict):
    """
    Dictionary with unlimited levels
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def condition(x,class_name):
    return x==class_name

def store_json(json_dict, predicted_trajectories_og, object_cls_og, seq_og,obj_id_og, prediction_length):
    # TODO: Improve trajectory samples!!!! (At this moment, it is the same both for '0' and '1')
    trajectory_samples = ['0','1']
    prob = 1.0
    na = 32 # Number of agents
    batch_size = int(predicted_trajectories_og.shape[1]/na)

    # print(">>>>>>>>>>>>>")

    # print("New batch")

    # print("keys 1: ", json_dict['10'].keys())

    for element in range(batch_size):
        object_cls = object_cls_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 32 -> 1 x 32
        seq = seq_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 2 -> 1 x 2
        obj_id = obj_id_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 32 -> 1 x 32 
        pred_fake_trajectories = predicted_trajectories_og[:,na*element:na*(element+1),:] # 12 x batch_size*na (8x32) x 2 -> 12 x 32 x 2
        
        object_indexes = []

        # if seq[0,0] == 7001 and seq[0,1] == 800:
        #     print("obj id: ", obj_id)
        #     print("cls: ", object_cls)
        #     print("....................")

        xx = str(int(seq[0,0]/1000))
        yyyy = str(int(seq[0,0]%1000))

        seq_name = ""

        if xx == '10':
            seq_name = "Town"+xx.zfill(2)+"HD"+"_seq"+yyyy.zfill(4)
        else:
            seq_name = "Town"+xx.zfill(2)+"_seq"+yyyy.zfill(4)

        seq_frame = str(int(seq[0,1]))
        # print("..................")
        for key,value in classes.items():
            if value == -1: # Dummy class
                continue
            indexes = np.where(np.array([condition(xi,value) for xi in object_cls]))[1]

            agent_dict = {}
            # print("Indexes: ", indexes)
            for i in range(pred_fake_trajectories.shape[1]): 
                if i in indexes:
                    ground_pos = []
                    for j in range(pred_fake_trajectories.shape[0]): 
                        if j<10:
                            ground_pos.append(pred_fake_trajectories[j,i,:].tolist()) # x,y
                    aux_dict = {}
                    aux_dict['state'] = ground_pos
                    aux_dict['prob'] = prob
                    agent_dict[str(int(obj_id[0,i]))] = aux_dict

            keys = json_dict[str(prediction_length)].keys()
            if key in keys:
                if json_dict[str(prediction_length)][key][seq_name][seq_frame][trajectory_samples[0]]: # Not empty
                    # print("keys 3: ", json_dict['10'].keys())
                    # print("Comb: ", prediction_length, key, seq_name, seq_frame, trajectory_sample)
                    # print("To add keys: ", agent_dict.keys())
                    previous_agent_dict = json_dict[str(prediction_length)][key][seq_name][seq_frame][trajectory_samples[0]]
                    # print("Prev keys: ", previous_agent_dict.keys())
                    previous_agent_dict.update(agent_dict)
                    # print("New keys: ", previous_agent_dict.keys())
                    agent_dict = previous_agent_dict
            # print(".............")
            # if key=="Mot" or key=="Ped" or key=="Cyc":
            #     print("agent dict: ", agent_dict)

            if agent_dict: # Not empty
                # print("key: ", key)
                # print("me meto")
                # print("agent dict: ", agent_dict, type(agent_dict))
                # print("keys 4: ", json_dict['10'].keys())
                for trajectory_sample in trajectory_samples:
                    json_dict[str(prediction_length)][key][seq_name][seq_frame][trajectory_sample] = agent_dict
    return json_dict

def get_generator(checkpoint, config):
    config.sophie.generator.decoder.linear_3.input_dim = config.dataset.batch_size*2*config.sophie.generator.social_attention.linear_decoder.out_features
    config.sophie.generator.decoder.linear_3.output_dim = config.dataset.batch_size*config.number_agents
    generator = SoPhieGenerator(config.sophie.generator)
    generator.build()
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

def evaluate(args, loader, generator, num_samples, pred_len, results_path, results_file, test_submission=False, skip=1):
    init_json = False
    json_dict = AutoTree()
    prediction_length = 10*skip # 10 (1 s), 20 (2 s), 50 (5 s)

    print("\n\n")

    print("Prediction length: ", prediction_length)

    final_ade, final_fde = 0,0
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            print("Evaluating batch: ", batch_index)
            batch = [tensor.cuda() for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_real, non_linear_ped, loss_mask, seq_start_end,
             _, frames, object_cls, seq, obj_id) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    frames, obs_traj
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                json_dict = store_json(json_dict,pred_traj_fake,object_cls,seq,obj_id,prediction_length)
                
                # print("json dict: ", json_dict)

                if not test_submission:
                    ade.append(displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    ))
            if not test_submission:
                ade_sum = evaluate_helper(ade, seq_start_end)
                fde_sum = evaluate_helper(fde, seq_start_end)

                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)

        if not test_submission:
            final_ade = sum(ade_outer) / (total_traj * pred_len)
            final_fde = sum(fde_outer) / (total_traj)
        else:
            final_ade = -1
            final_fde = -1

    print("Finish evaluation")
    object_class = list(json_dict[str(prediction_length)].keys())[0]
    print("Object class: ", object_class)
    
    final_results = os.path.join(results_path, results_file + '.json')
    if os.path.isfile(final_results):
        print("The file does exist")
        with open(final_results) as f:
            previous_dict = json.load(f)
            previous_dict = AutoTree(previous_dict)
            previous_lengths = list(previous_dict.keys())
            try:
                classes_length_previous_dict = list(previous_dict[str(prediction_length)].keys())
            except:
                classes_length_previous_dict = []
            print("Previous lengths: ", previous_lengths)
            print(f"Classes prediction length {prediction_length}: ", classes_length_previous_dict)
            if str(prediction_length) in previous_lengths and object_class in classes_length_previous_dict:
                print(f"Update prediction_length {prediction_length} and class {object_class}")
            else:
                print(f"New prediction_length {prediction_length} and class {object_class}")
            previous_dict[str(prediction_length)][object_class] = json_dict[str(prediction_length)][object_class]
            json_dict = previous_dict

    with open(final_results, 'w') as fp:
        json.dump(json_dict, fp)

    return final_ade, final_fde

def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    BASE_DIR = Path(__file__).resolve().parent

    with open(r'./configs/sophie_aiodrive.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR

    # windows_frames = [30,180,330,480,630,780,930] # Validating test
    windows_frames = [42,192,342,492,642,792,942] # Final submission (test)
    test_submission = config_file.dataset.test_submission
    skip = args.skip # skip = 1 (prediction 1 s) ; skip = 2 (prediction 2 s) ....
    obs_len = 8
    pred_len = 12 
    if test_submission:
        pred_len = 0 # 0 only in test, since we do not have these data

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint.config_cp, config_file)
        test_path = os.path.join(config_file.base_dir, args.dataset_path, "test")

        if config_file.dataset.absolute_route:
            videos_path = config_file.dataset.video
        else:
            videos_path = os.path.join(config_file.base_dir, config_file.dataset.video)

        print("videos_path: ", videos_path)
        
        data_test = AioDriveDataset(test_path, videos_path=videos_path, 
                                    windows_frames=windows_frames, skip=skip, 
                                    obs_len=obs_len, pred_len=pred_len, phase='testing')
        pred_len = data_test.pred_len

        test_loader = DataLoader(
            data_test,
            batch_size=config_file.dataset.batch_size,
            shuffle=config_file.dataset.shuffle,
            num_workers=config_file.dataset.num_workers,
            collate_fn=seq_collate_image_aiodrive)

        print("\n\nresults file: ", args.results_file)

        ade, fde = evaluate(checkpoint.config_cp, test_loader, 
                            generator, args.num_samples, pred_len, 
                            args.results_path, args.results_file,
                            test_submission, skip=skip)

        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            args.dataset_path, pred_len, ade, fde))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
