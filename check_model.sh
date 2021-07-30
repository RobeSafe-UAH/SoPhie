#!/usr/bin/env bash 

model_path="test_with_model.pt"  # $1
results_file="final_submission_v50" # $2

# Car

python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=1
python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=2
python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=5

# Cyclist 

# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=1
# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=2
# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=5

# Motorcycle

# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=1
# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=2
# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=5

# Pedestrian 

# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=1
# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=2
# python3.8 evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=5
