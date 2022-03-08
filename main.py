import yaml
import json
import logging
import os
import sys
import pdb
import argparse

from datetime import datetime
from prodict import Prodict
from pathlib import Path

TRAINER_LIST = ["so", "sovi", "trans_so", "trans_sovi"]

def create_logger(file_path):
    FORMAT = '[%(levelname)s: %(lineno)4d]: %(message)s'
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    # formatter = logging.Formatter('[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s')

    # stdout_handler = logging.StreamHandler(sys.stdout)
    # stdout_handler.setLevel(logging.DEBUG)
    # stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stdout_handler)
    return logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, type=str, choices=TRAINER_LIST)
    args = parser.parse_args()
    print(args.trainer)
    
    trainer = None
    if args.trainer == "so":
        from sophie.trainers.trainer_gen_so import model_trainer
    elif args.trainer == "sovi":
        from sophie.trainers.trainer_gen_sovi import model_trainer
    elif args.trainer == "trans_so":
        from sophie.trainers.trainer_gen_trans_so import model_trainer
    elif args.trainer == "trans_sovi":
        from sophie.trainers.trainer_gen_trans_sovi import model_trainer
    else:
        assert 1==0, "Error"

    trainer = model_trainer

    BASE_DIR = Path(__file__).resolve().parent

    print("BASE_DIR: ", BASE_DIR)

    with open(r'./configs/sophie_argoverse.yml') as config_file:
        config_file = yaml.safe_load(config_file)

        # Fill some additional dimensions

        past_observations = config_file["hyperparameters"]["obs_len"]
        num_agents_per_obs = config_file["hyperparameters"]["num_agents_per_obs"]
        config_file["sophie"]["generator"]["social_attention"]["linear_decoder"]["out_features"] = past_observations * num_agents_per_obs
        
        config_file["base_dir"] = BASE_DIR
        exp_path = os.path.join(config_file["base_dir"], config_file["hyperparameters"]["output_dir"])   
        route_path = exp_path + "/config_file.yml"

        if not os.path.exists(exp_path):
            print("Create experiment path: ", exp_path)
            os.mkdir(exp_path)

        with open(route_path,'w') as yaml_file:
            yaml.dump(config_file, yaml_file, default_flow_style=False)

        config_file = Prodict.from_dict(config_file)

    now = datetime.now()
    time = now.strftime("%H:%M:%S")

    logger = create_logger(os.path.join(exp_path, f"{config_file.dataset_name}_{time}.log"))
    
    trainer(config_file, logger)