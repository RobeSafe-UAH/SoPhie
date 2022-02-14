import yaml
import json
import logging
import os
import sys

from datetime import datetime
from prodict import Prodict
from pathlib import Path

# from sophie.trainers.trainer import model_trainer
# from sophie.trainers.trainer_gan import model_trainer
from sophie.trainers.trainer_sophie_adaptation import model_trainer

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
    
    BASE_DIR = Path(__file__).resolve().parent

    print("BASE_DIR: ", BASE_DIR)

    with open(r'./configs/sophie_argoverse.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        print(yaml.dump(config_file, default_flow_style=False))
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR

    # Fill some additional dimensions

    past_observations = config_file.hyperparameters.obs_len
    num_agents_per_obs = config_file.hyperparameters.num_agents_per_obs
    config_file.sophie.generator.social_attention.linear_decoder.out_features = past_observations * num_agents_per_obs

    exp_path = os.path.join(
                    config_file.base_dir, config_file.hyperparameters.output_dir
                )

    if not os.path.exists(exp_path):
    #     # raise Exception(f"Experiment path does not exist: {exp_path}")
    #     logger.error(f"Experiment path does not exist: {exp_path}")
        print("Experiment Path does not exist: ", exp_path)
        sys.exit(1)
    now = datetime.now()
    time = now.strftime("%H:%M:%S")

    logger = create_logger(os.path.join(exp_path, f"{config_file.dataset_name}_{time}.log"))
    
    model_trainer(config_file, logger)