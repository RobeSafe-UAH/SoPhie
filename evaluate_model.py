import argparse
import os
import yaml
import torch
from pathlib import Path
from prodict import Prodict
from torch.utils.data import DataLoader

from attrdict import AttrDict

from sophie.data_loader.ethucy.dataset import read_file, EthUcyDataset, seq_collate_image
from sophie.models import SoPhieGenerator
from sophie.modules.evaluation_metrics import displacement_error, final_displacement_error
from sophie.utils.utils import relative_to_abs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

class AutoTree(dict):
    """
    Dictionary with unlimited levels
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def get_generator(checkpoint, config):
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

def evaluate(args, loader, generator, num_samples, pred_len):
    init_json = False
    json_dict = AutoTree()

    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, frames, prediction_length, object_class, seq_name, 
             seq_frame, object_id) = batch

            json_dict[prediction_length][object_class][seq_name][seq_frame] = []

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            agent_dict = {}

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    frames, obs_traj
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                for i in range(pred_traj_fake.shape[1]): 
                    ground_pos = []
                    for j in range(pred_traj_fake.shape[0]): 
                        aux = []
                        aux.append(pred_traj_fake[j,i,:]) # x,y 
                        ground_pos.append(ground_pos)
                    # We assume here i is the identifier
                    aux_dict = {}
                    aux_dict['state'] = ground_pos
                    aux_dict['prob'] = prob
                    agent_dict[str(object_id[i])] = aux_dict

                # Create dict to json 

                json_dict[prediction_length][object_class][seq_name][seq_frame].append(agent_dict)

                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * pred_len)
        fde = sum(fde_outer) / (total_traj)
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

    with open(r'./experiments/1/sophie.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint.config_cp, config_file)
        test_path = os.path.join(config_file.base_dir, config_file.dataset.path, "test")
        data_test = EthUcyDataset(test_path, videos_path=os.path.join(config_file.base_dir, config_file.dataset.video))
        pred_len = data_test.pred_len

        test_loader = DataLoader(
            data_test,
            batch_size=config_file.dataset.batch_size,
            shuffle=config_file.dataset.shuffle,
            num_workers=config_file.dataset.num_workers,
            collate_fn=seq_collate_image)
        ade, fde = evaluate(checkpoint.config_cp, test_loader, generator, args.num_samples, pred_len)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            config_file.dataset.path, pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
