import yaml

from prodict import Prodict
from pathlib import Path

from sophie.trainers.trainer import model_trainer

if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent

    print("BASE_DIR: ", BASE_DIR)

    with open(r'./configs/sophie_argoverse.yml') as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)
        config_file.base_dir = BASE_DIR

    print("Config file: ", config_file)
    # model_trainer(config_file)