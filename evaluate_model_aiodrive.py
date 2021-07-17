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
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--results_file', default='test_json', type=str)

classes = {"Car":0, "Cyc":1, "Mot":2, "Ped":3, "Dum":4} # Car, Ped, Mot, Cyc, Dummy

class AutoTree(dict):
    """
    Dictionary with unlimited levels
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def condition(x,class_name):
    return x==class_name

def store_json(json_dict,predicted_trajectories_og,object_cls_og,seq_og,obj_id_og):
    prediction_length = '10'
    trajectory_sample = '0'
    prob = 1.0
    na = 32 # Number of agents
    batch_size = int(predicted_trajectories_og.shape[1]/na)

    for element in range(batch_size):
        object_cls = object_cls_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 32 -> 1 x 32
        seq = seq_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 2 -> 1 x 2
        obj_id = obj_id_og[element].reshape(1,-1).cpu().data.numpy() # batch_size x 32 -> 1 x 32 
        pred_fake_trajectories = predicted_trajectories_og[:,na*element:na*(element+1),:] # 12 x batch_size*na (8x32) x 2 -> 12 x 32 x 2

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
                        if j<10:
                            ground_pos.append(pred_fake_trajectories[j,i,:].tolist()) # x,y
                    # We assume here i is the identifier
                    # print("Ground pos: ", ground_pos)
                    aux_dict = {}
                    aux_dict['state'] = ground_pos
                    aux_dict['prob'] = prob
                    agent_dict[str(int(obj_id[0,i]))] = aux_dict
                    json_dict[prediction_length][key][seq_name][seq_frame][trajectory_sample] = agent_dict
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

def evaluate(args, loader, generator, num_samples, pred_len, results_file='test_json', final_submission=False):
    init_json = False
    json_dict = AutoTree()

    final_ade, final_fde = 0,0
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            print("Evaluating ...")
            batch = [tensor.cuda() for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_real, non_linear_ped, loss_mask, seq_start_end,
             _, frames, object_cls, seq, obj_id) = batch

            print("Frames: ", frames, frames.shape)
            print("Classes: ", object_cls, object_cls.shape)
            print("Seq: ", seq, seq.shape)
            print("Obj id: ", obj_id, obj_id.shape)
            print("Obs traj: ", obs_traj, obs_traj.shape)

            assert 1 == 0

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    frames, obs_traj
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                json_dict = store_json(json_dict,pred_traj_fake,object_cls,seq,obj_id)
                # print("json dict: ", json_dict)

                if not final_submission:
                    ade.append(displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    ))
            if not final_submission:
                ade_sum = evaluate_helper(ade, seq_start_end)
                fde_sum = evaluate_helper(fde, seq_start_end)

                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)

        if not final_submission:
            final_ade = sum(ade_outer) / (total_traj * pred_len)
            final_fde = sum(fde_outer) / (total_traj)
        else:
            final_ade = -1
            final_fde = -1

    print("Finish evaluation")
    final_results = results_file + '.json'
    print("Final results file: ", final_results)
    with open(final_results, 'w') as fp:
        json.dump(json_dict, fp)

    return ade, fde

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

    windows_frames = [30,180,330,480,630,780,930] # Validating test
    final_submission = config_file.dataset.final_submission

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint.config_cp, config_file)
        test_path = os.path.join(config_file.base_dir, config_file.dataset.path, "test")

        if config_file.dataset.absolute_route:
            videos_path = config_file.dataset.video
        else:
            videos_path = os.path.join(config_file.base_dir, config_file.dataset.video)

        print("videos_path: ", videos_path)
        
        # windows_frames = [42,192,342,492,642,792,942] # Final submission
        # skip = 1 (prediction 1 s) ; skip = 2 (prediction 2 s) ....
        data_test = AioDriveDataset(test_path, videos_path=videos_path, 
                                    windows_frames=windows_frames, skip=1, 
                                    obs_len=8, pred_len=0)
        pred_len = data_test.pred_len

        test_loader = DataLoader(
            data_test,
            batch_size=config_file.dataset.batch_size,
            shuffle=config_file.dataset.shuffle,
            num_workers=config_file.dataset.num_workers,
            collate_fn=seq_collate_image_aiodrive)

        ade, fde = evaluate(checkpoint.config_cp, test_loader, 
                            generator, args.num_samples, pred_len, 
                            args.results_file, final_submission)

        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            config_file.dataset.path, pred_len, ade, fde))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
