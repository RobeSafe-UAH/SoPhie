import yaml
import torch
import numpy as np
from pathlib import Path
from prodict import Prodict
import csv
import pdb
import sys
import os

from sklearn import linear_model
from skimage.measure import LineModelND, ransac

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BASE_DIR = "/home/robesafe/tesis/SoPhie"
sys.path.append(BASE_DIR)

from sophie.utils.utils import relative_to_abs_sgan
# from sophie.models.sophie_adaptation import TrajectoryGenerator
from sophie.models.mp_so import TrajectoryGenerator
from sophie.data_loader.argoverse.dataset_sgan_version import ArgoverseMotionForecastingDataset, seq_collate
# from sophie.trainers.trainer_sophie_adaptation_single_agent import cal_ade, cal_fde
from sophie.trainers.trainer_sophie_adaptation import cal_ade, cal_fde

pred_gt_file = "test_trajectories/" "pred_gt.npy"
pred_fake_file = "test_trajectories/" "pred_fake.npy"
linear_obj_file = "test_trajectories/" "linear_obj.npy"
non_linear_obj_file = "test_trajectories/" "non_linear_obj.npy"
mask_file = "test_trajectories/" "mask.npy"

pred_len = 30

try:
    assert 1 == 0
    with open(pred_gt_file, 'rb') as my_file:
        agent_traj_gt =  torch.from_numpy(np.load(my_file))

    with open(pred_fake_file, 'rb') as my_file:
        agent_traj_fake =  torch.from_numpy(np.load(my_file))

    with open(linear_obj_file, 'rb') as my_file:
        linear_obj =  torch.from_numpy(np.load(my_file))

    with open(non_linear_obj_file, 'rb') as my_file:
        non_linear_obj =  torch.from_numpy(np.load(my_file))

    with open(mask_file, 'rb') as my_file:
        mask =  torch.from_numpy(np.load(my_file))

    print("agent gt: ", agent_traj_gt.shape)
    print("agent fake: ", agent_traj_fake.shape)
    print("agent linear obj: ", linear_obj.shape)
    print("agent non_linear_obj: ", non_linear_obj.shape)
    print("agent mask: ", mask.shape)

    ade, ade_l, ade_nl = cal_ade(
        agent_traj_gt, agent_traj_fake, linear_obj, non_linear_obj, mask
    )

    print("ade: ", ade.item()/pred_len)

    fde, fde_l, fde_nl = cal_fde(
        agent_traj_gt, agent_traj_fake, linear_obj, non_linear_obj, mask
    )

    print("fde: ", fde.item())

except:
    with open(r'./configs/sophie_argoverse.yml') as config:
        config = yaml.safe_load(config)
        config = Prodict.from_dict(config)
        config.base_dir = BASE_DIR

    # Fill some additional dimensions

    past_observations = config.hyperparameters.obs_len
    num_agents_per_obs = config.hyperparameters.num_agents_per_obs
    config.sophie.generator.social_attention.linear_decoder.out_features = past_observations * num_agents_per_obs

    split_percentage = 0.005
    batch_size = 1

    print("dataset path: ", config.dataset.path)

    data_val = ArgoverseMotionForecastingDataset(dataset_name=config.dataset_name,
                                                 root_folder=config.dataset.path,
                                                 obs_len=config.hyperparameters.obs_len,
                                                 pred_len=config.hyperparameters.pred_len,
                                                 distance_threshold=config.hyperparameters.distance_threshold,
                                                 split="val",
                                                 num_agents_per_obs=config.hyperparameters.num_agents_per_obs,
                                                 split_percentage=split_percentage,
                                                 shuffle=config.dataset.shuffle,
                                                 class_balance=config.dataset.class_balance,
                                                 obs_origin=config.hyperparameters.obs_origin)

    loader = DataLoader(data_val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=config.dataset.num_workers,
                            collate_fn=seq_collate)

    exp_name = "gen_exp/exp7"
    model_path = BASE_DIR + "/save/argoverse/" + exp_name + "/argoverse_motion_forecasting_dataset_0_with_model.pt"
    checkpoint = torch.load(model_path)
    generator = TrajectoryGenerator(config.sophie.generator)

    generator.load_state_dict(checkpoint.config_cp['g_best_state'])
    generator.cuda() # Use GPU
    generator.eval()

    num_samples = 5
    output_all = []

    ade_list = []
    fde_list = []

    num_samples = 6
    output_all = []

    ade_list = []
    fde_list = []

    DEBUG_TRAJECTORY_CLASSIFIER = False

    # TODO: Implement top-k trajectory

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index > 99:
                break
            batch = [tensor.cuda() for tensor in batch]
            
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_obj,
             loss_mask, seq_start_end, frames, object_cls, obj_id, ego_origin, _,_) = batch

            predicted_traj = []
            agent_idx = torch.where(object_cls==1)[0].cpu().numpy()
            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)

            # Check if the trajectory is a straight line or has a curve
  
            agent_seq = traj_real[:,agent_idx,:].cpu().detach().numpy() # 50 x batch_size x 2
            agent_x = agent_seq[:,0,0].reshape(-1,1) # We assume here 50 x 1 x 2 (1 represents batch_size = 1)
            agent_y = agent_seq[:,0,1].reshape(-1,1)

            ## Sklearn    

            ransac = linear_model.RANSACRegressor(residual_threshold=2)
            ransac.fit(agent_x,agent_y)

            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            num_inliers = len(np.where(inlier_mask == True)[0])

            ## Study consecutive inliers

            cnt = 0
            num_min_outliers = 8 # Minimum number of consecutive outliers to consider the trajectory as curve
            is_curve = False
            for is_inlier in inlier_mask:
                if not is_inlier:
                    cnt += 1
                else:
                    cnt = 0

                if cnt >= num_min_outliers:
                    is_curve = True

            if DEBUG_TRAJECTORY_CLASSIFIER:
                x_max = agent_x.max()
                x_min = agent_x.min()
                num_steps = 20
                step_dist = (x_max - x_min) / num_steps
                line_x = np.arange(x_min, x_max, step_dist)[:, np.newaxis]
                line_y_ransac = ransac.predict(line_x)

                y_min = line_y_ransac.min()
                y_max = line_y_ransac.max()

                lw = 2
                plt.scatter(
                    agent_x[inlier_mask], agent_y[inlier_mask], color="blue", marker=".", label="Inliers"
                )
                plt.scatter(
                    agent_x[outlier_mask], agent_y[outlier_mask], color="red", marker=".", label="Outliers"
                )

                plt.plot(
                    line_x,
                    line_y_ransac,
                    color="cornflowerblue",
                    linewidth=lw,
                    label="RANSAC regressor",
                )
                plt.legend(loc="lower right")
                plt.xlabel("X (m)")
                plt.ylabel("Y (m)")
                plt.title('Sequence {}. Num inliers: {}. Is a curve: {}'.format(batch_index,num_inliers,is_curve))

                threshold = 15
                plt.xlim([x_min-threshold, x_max+threshold])
                plt.ylim([y_min-threshold, y_max+threshold])

                plt.show()
     
            top_k_ade = 50000
            top_k_fde = 50000

            for _ in range(num_samples):
                # Get predictions
                # pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, frames, agent_idx) # seq_start_end)
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, agent_idx)

                # Get predictions in absolute coordinates
                pred_traj_fake = relative_to_abs_sgan(pred_traj_fake_rel, obs_traj[-1,agent_idx, :]) # 30,1,2
                traj_fake = torch.cat([obs_traj[:,agent_idx, :], pred_traj_fake], dim=0) # 50,1,2
                predicted_traj.append(traj_fake)

                # Get metrics
                agent_traj_gt = pred_traj_gt[:,agent_idx,:] # From Multi to Single (30 x bs x 2)
                agent_traj_fake = pred_traj_fake # The output of the model is already single (30 x bs x 2)

                
                agent_obj_id = obj_id[agent_idx]
                agent_non_linear_obj = non_linear_obj[agent_idx]

                agent_mask = np.where(agent_obj_id.cpu() == -1, 0, 1)
                agent_mask = torch.tensor(agent_mask, device=agent_obj_id.device).reshape(-1)
                
                agent_linear_obj = 1 - agent_non_linear_obj

                # with open(pred_gt_file, 'wb') as my_file:
                #     np.save(my_file, agent_traj_gt.cpu().detach().numpy())
                
                # with open(pred_fake_file, 'wb') as my_file:
                #     np.save(my_file, agent_traj_fake.cpu().detach().numpy())
                
                # with open(linear_obj_file, 'wb') as my_file:
                #     np.save(my_file, agent_linear_obj.cpu().detach().numpy())
               
                # with open(non_linear_obj_file, 'wb') as my_file:
                #     np.save(my_file, agent_non_linear_obj.cpu().detach().numpy())

                # with open(mask_file, 'wb') as my_file:
                #     np.save(my_file, agent_mask.cpu().detach().numpy())

                ade, ade_l, ade_nl = cal_ade(
                    agent_traj_gt, agent_traj_fake, agent_linear_obj, agent_non_linear_obj, agent_mask
                )

                fde, fde_l, fde_nl = cal_fde(
                    agent_traj_gt, agent_traj_fake, agent_linear_obj, agent_non_linear_obj, agent_mask
                )

                ade = ade.item() / pred_len
                fde = fde.item()

                if ade < top_k_ade:
                    top_k_ade = ade
                    top_k_fde = fde

            ade_list.append(ade)
            fde_list.append(fde)

            predicted_traj = torch.stack(predicted_traj, axis=0)
            predicted_traj = predicted_traj.cpu().numpy()
            output_all.append(predicted_traj)

        ade = round(sum(ade_list) / (len(ade_list)),3)
        print("ade: ", ade)
        fde = round(sum(fde_list) / (len(fde_list)),3)
        print("fde: ", fde)

        # Write ade and fde values in CSV

        file_dir = str(Path(__file__).resolve().parent)

        with open(file_dir+'/test_trajectories/'+exp_name+'_metrics_val.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            header = ['Sequence','ADE','FDE']
            csv_writer.writerow(header)

            sorted_indeces = np.argsort(ade_list)

            for _,i in enumerate(sorted_indeces):
                seq_id = i
                curr_ade = round(ade_list[i],3)
                curr_fde = round(fde_list[i],3)
                data = [str(i),curr_ade,curr_fde]

                csv_writer.writerow(data)

            # Write mean ADE and FDE 

            csv_writer.writerow(['-','-','-'])
            csv_writer.writerow(['Mean',ade,fde])

    

