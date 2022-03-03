#!/usr/bin/env bash 

evaluate_model="evaluate/evaluate_model_aiodrive.py"
model_path="save/aiodrive/test_with_model_95_percent_labels.pt" 
results_path="results/aiodrive"
results_file="submission_95_percent_training_all_objects_all_skips" 

python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=1

# Car

# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_path= --results_file=$results_file --skip=1
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=2
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Car/ --results_file=$results_file --skip=5

# # Cyclist 

# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=1
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=2
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Cyc/ --results_file=$results_file --skip=5

# # Motorcycle

# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=1
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=2
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Mot/ --results_file=$results_file --skip=5

# # Pedestrian 

# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=1
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=2
# python3.8 $evaluate_model --model_path=$model_path --dataset_path=data/datasets/aiodrive/aiodrive_Ped/ --results_file=$results_file --skip=5
