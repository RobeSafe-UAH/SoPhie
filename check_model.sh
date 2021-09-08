#!/usr/bin/env bash 

evaluate_model="evaluate/evaluate_model_aiodrive.py"
model_path="save/test_with_model_95_percent_labels.pt"  # $1
results_file="final_submission_v50" # $2

# Car

python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=1
# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=2
# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=5

# # Cyclist 

# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=1
# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=2
# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=5

# # Motorcycle

# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=1
# python3.8 evaluateevaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=2
# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=5

# # Pedestrian 

# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=1
# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=2
# python3.8 evaluate/evaluate_model_aiodrive.py --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=5
