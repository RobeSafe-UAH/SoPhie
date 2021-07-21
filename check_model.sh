#!/usr/bin/env bash 

python3.8 evaluate_model_aiodrive.py --model_path=test_with_model.pt --results_file=final_submission_v2 --skip=1
python3.8 evaluate_model_aiodrive.py --model_path=test_with_model.pt --results_file=final_submission_v2 --skip=2
python3.8 evaluate_model_aiodrive.py --model_path=test_with_model.pt --results_file=final_submission_v2 --skip=5

