from sophie.trainers.trainer import model_trainer
import yaml
from prodict import Prodict

if __name__ == "__main__":
    
    with open("/home/robesafe/git/SoPhie/configs/sophie.yml") as config_file:
        config_file = yaml.safe_load(config_file)
        config_file = Prodict.from_dict(config_file)
    model_trainer(config_file)