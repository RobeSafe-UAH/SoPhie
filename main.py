import yaml
import json
import logging

from prodict import Prodict
from pathlib import Path

from sophie.trainers.trainer import model_trainer

if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent

    print("BASE_DIR: ", BASE_DIR)

    with open(r'./configs/sophie_argoverse.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        print("\n")
        print(yaml.dump(config_file, default_flow_style=False))
        print("\n")
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR

    # Fill some additional dimensions

    past_observations = config_file.hyperparameters.obs_len
    num_agents = config_file.hyperparameters.number_agents
    config_file.sophie.generator.social_attention.linear_decoder.out_features = past_observations * num_agents
    
    model_trainer(config_file)