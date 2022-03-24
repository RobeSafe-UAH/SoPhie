#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Sun Mar 06 23:47:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

import numpy as np
import pdb
import cv2
from numpy.random import default_rng
from sklearn import linear_model
import math
import random
import pdb
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
from random import sample
import copy
import torch

def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    try:
        x,y,z = b.x, b.y, 0.0
        X,Y,Z = e.x, e.y, 0.0
    except:
        x,y,z = b
        X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v

    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    try:   
        X,Y,Z = w.x, w.y, 0.0
    except:
        X,Y,Z = w
    return (x+X, y+Y, z+Z)

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)

    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def get_non_linear(file_id, curr_seq, idx=0, obj_kind=2, threshold=2, debug_trajectory_classifier=False):
    """
    Non-linear means the trajectory (of the AGENT in the present case) is a curve. 
    Otherwise, it is considered as a linear (straight) trajectory
    """

    agent_seq = curr_seq[idx,:,:] #.cpu().detach().numpy()
    num_points = curr_seq.shape[2]

    agent_x = agent_seq[0,:].reshape(-1,1)
    agent_y = agent_seq[1,:].reshape(-1,1)

    # Fit a RANSAC regressor  

    ransac = linear_model.RANSACRegressor(residual_threshold=threshold, 
                                          max_trials=30, 
                                          min_samples=round(0.6*num_points))
    ransac.fit(agent_x,agent_y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    num_inliers = len(np.where(inlier_mask == True)[0])

    ## Study consecutive outliers

    cnt = 0
    num_min_outliers = 8 # Minimum number of consecutive outliers to consider the trajectory as curve
    non_linear = 0.0
    reason = ""
    for is_inlier in inlier_mask:
        if not is_inlier:
            cnt += 1
        else:
            cnt = 0

        if cnt >= num_min_outliers:
            non_linear = 1.0
            reason = "ransac"
            break

    # Study distance from intermediate points to the line that links the first and last point

    first_point = (agent_x[0], agent_y[0], 0)
    last_point = (agent_x[-1], agent_y[-1], 0)
    num_out = 0
    num_far_out = 0
    num_close_out = 0
    flag = False

    for index in range(num_points):
        point = (agent_x[index], agent_y[index], 0)
        dist,_ = pnt2line(point,first_point,last_point)
        # print("index, dist: ", index, dist)
        if dist >= threshold:  
            num_out += 1
            if num_out >= num_min_outliers:
                flag = True
                reason = reason + " + normal outliers"
                break

            if dist >= threshold*1.5:
                num_far_out += 1
                if num_far_out >= round(0.5 * num_min_outliers):
                    flag = True
                    reason = reason + " + far outliers"
                    break
        if dist >= round(0.66*threshold):
            num_close_out += 1
            if num_close_out >= round(1.2 * num_min_outliers):
                flag = True
                reason = reason + " + close outliers"
                break
        else:
            num_out = 0
            num_far_out
            num_close_out = 0

    # coef, res, _, _, _ = np.polyfit(agent_x.flatten(),agent_y.flatten(),1,full=True)
    # coef, res_list = np.polynomial.hermite.hermfit(agent_x.flatten(),agent_y.flatten(),2,full=True)
    # res = res_list[0]

    if non_linear or flag:
        non_linear = 1.0
    else:
        non_linear = 0.0

    if debug_trajectory_classifier:
        x_max = agent_x.max()
        x_min = agent_x.min()
        num_steps = 20
        step_dist = (x_max - x_min) / num_steps
        line_x = np.arange(x_min, x_max, step_dist)[:, np.newaxis]
        line_y_ransac = ransac.predict(line_x)

        y_min = line_y_ransac.min()
        y_max = line_y_ransac.max()
        
        plt.scatter(
            agent_x[inlier_mask], agent_y[inlier_mask], color="blue", marker=".", label="Inliers"
        )
        plt.scatter(
            agent_x[outlier_mask], agent_y[outlier_mask], color="red", marker=".", label="Outliers"
        )
        plt.scatter(agent_x[0], agent_y[0], s=200, color='green', marker=".", label="First obs")

        lw = 2
        plt.plot(
            line_x,
            line_y_ransac,
            color="cornflowerblue",
            linewidth=lw,
            label="RANSAC regressor",
        )

        # yfit = np.polyval(coef,line_x)
        # plt.plot(line_x,yfit, label='fit')

        plt.legend(loc="lower right")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if non_linear == 1.0:
            non_linear = True
        else:
            non_linear = False

        if obj_kind == 0:
            obj = "AV"
        elif obj_kind == 1:
            obj = "AGENT"
        else:
            obj = "OTHER"

        plt.title('Sequence {}. Obstacle: {}. Num inliers: {}. Is a curve: {}. Reason: {}'.format(file_id,obj,num_inliers,non_linear,reason))

        threshold = 5
        plt.xlim([x_min-threshold, x_max+threshold])
        plt.ylim([y_min-threshold, y_max+threshold])

        plt.show()
    return non_linear

# Data augmentation functions

def get_data_aug_combinations(num_augs):
    """
    Tuple to enable (1) a given augmentation
    """

    return np.random.randint(2,size=num_augs).tolist()

def get_pairs(percentage,num_obs,start_from=1):
    """
    Return round(percentage*num_obs) non-consecutive indeces in the range(start_from,num_obs-1)
    N.B. Considering this algorithm, the maximum number to calculate non-consecutive indeces
    is 40 % (0.4)
    
    Input:
        - percentage
        - num_obs
        - start_from
    Output:
        - indeces
    """

    indeces = set()
    while len(indeces) !=  round(percentage*num_obs):
        a = random.randint(start_from,num_obs-1) # From second point (1) to num_obs - 1
        if not {a-1,a,a+1} & indeces: # Set intersection: empty == False == no common numbers
            indeces.add(a)

    indeces = sorted(indeces)
    return indeces

def swap_points(traj,num_obs=20,percentage=0.2,post=False):
    """
    Swapping (x(i) <-> x(i+1)) of observation data (Not the whole sequence)

    N.B. Points to be swapped cannot be consecutive to avoid:
    x(i),x(i+1),x(i+2) -> x(i+1),x(i),x(i+2) -> x(i+1),x(i+2),x(i) # TODO: Try this?

    Input:
        - traj: Whole trajectory (2 (x|y) x seq_len)
        - num_obs: Number of observations of the whole trajectory
        - percentage: Ratio of pairs to be swapped. Typically the number of pairs 
          to be swapped will be 0.2. E.g. N=4 if num_observations is 20
        - post: Flag to swap with the next (x(i+1)) or previous (x(i-1)) point. 
          By default: False (swap with the previous point)
    Output:
        - swapped_traj: Whole trajectory (2 (x|y) x seq_len) with swapped non-consecutive points in
          the range (1,num_obs-1)
    """

    swapped_traj = copy.deepcopy(traj)

    swapped_pairs = get_pairs(percentage,num_obs)

    for index_pair in swapped_pairs:
        aux = copy.copy(swapped_traj[:,index_pair])
        swapped_traj[:,index_pair] = swapped_traj[:,index_pair-1]
        swapped_traj[:,index_pair-1] = aux

    return swapped_traj

def erase_points(traj,num_obs=20,percentage=0.2,post=False):
    """
    Remove (x(i) and subsitute with x(i-1)) 
    E.g. x(0), x(1), x(2) -> x(0), x(0), x(2)
    Flag: Substitute with i-1 or i+1

    Input:
        - traj: Whole trajectory (2 (x|y) x seq_len)
        - num_obs: Number of observations of the whole trajectory
        - percentage: Ratio of pairs to be erased. Typically the number of pairs 
          to be swapped will be 0.2. E.g. N=4 if num_observations is 20
        - post: Flag to erase the current point and substitute with the next (x(i+1)) or 
          previous (x(i-1)) point. By default: False (substitute with the previous point)
    Output:
        - erased_traj: Whole trajectory (2 (x|y) x seq_len) with erased non-consecutive points in
          the range (1,num_obs-1)
    """

    erased_traj = copy.deepcopy(traj)

    erased_pairs = get_pairs(percentage,num_obs)

    for index_pair in erased_pairs:
        erased_traj[:,index_pair] = erased_traj[:,index_pair-1]

    return erased_traj

def erase_points_collate(traj,apply_dropout,num_obs=20,percentage=0.2,post=False):
    """
    Remove (x(i) and subsitute with x(i-1)) 
    E.g. x(0), x(1), x(2) -> x(0), x(0), x(2)
    Flag: Substitute with i-1 or i+1

    Input:
        - traj: 20 x num_agents x 2
        - num_obs: Number of observations of the whole trajectory
        - percentage: Ratio of pairs to be erased. Typically the number of pairs 
          to be swapped will be 0.2. E.g. N=4 if num_observations is 20
        - post: Flag to erase the current point and substitute with the next (x(i+1)) or 
          previous (x(i-1)) point. By default: False (substitute with the previous point)
    Output:
        - erased_traj: Whole trajectory (20 x num_agents x 2) with erased non-consecutive points in
          the range (1,num_obs-1)
    """

    erased_traj = copy.deepcopy(traj)
    erased_traj_aux = torch.zeros((erased_traj.shape))

    for i,app_dropout in enumerate(apply_dropout):
        
        _erased_traj = erased_traj[:,i,:]

        if app_dropout:
            erased_pairs = get_pairs(percentage,num_obs)
            for index_pair in erased_pairs:
                _erased_traj[index_pair,:] = _erased_traj[index_pair-1,:]

        erased_traj_aux[:,i,:] = _erased_traj

    return erased_traj_aux

## 3. Gaussian noise -> Add gaussian noise to the observation data

def add_gaussian_noise(traj,num_obs=20,multi_point=True,mu=0,sigma=0.5):
    """
    Input:
        - traj: 2 x 50
    Output: 
        - noise_traj: 2 x 50 with gaussian noise in the observation points

    If multi_point = False, apply a single x|y offset to all observation points.
    Otherwise, apply a particular x|y per observation point.

    By default, multi_point = True since it is more challenging.
    """

    noised_traj = copy.deepcopy(traj)

    if multi_point:
        size = num_obs
    else:
        size = 1

    x_offset, y_offset = np.random.normal(mu,sigma,size=size), np.random.normal(mu,sigma,size=size)

    noised_traj[0,:num_obs] += x_offset
    noised_traj[1,:num_obs] += y_offset

    return noised_traj

def add_gaussian_noise_collate(traj,apply_gaussian_noise,num_agents,num_obs=20,multi_point=True,mu=0,sigma=0.5):
    """
    Input:
        - traj: 20 x num_agents x 2
    Output: 
        - noise_traj: 20 x num_agents x 2 with gaussian noise in the observation points

    If multi_point = False, apply a single x|y offset to all observation points.
    Otherwise, apply a particular x|y per observation point.

    By default, multi_point = True since it is more challenging.
    """

    noised_traj = copy.deepcopy(traj)

    if multi_point:
        size = (num_obs,num_agents)
    else:
        size = (1,num_agents)
    
    x_offset, y_offset = np.random.normal(mu,sigma,size=size), np.random.normal(mu,sigma,size=size) # TODO: Do this with PyTorch

    # Not apply in the following objects

    indeces = np.where(apply_gaussian_noise == 0)
    x_offset[:,indeces] = 0
    y_offset[:,indeces] = 0

    noised_traj[:,:,0] += x_offset
    noised_traj[:,:,1] += y_offset

    return noised_traj

## 4. Rotate trajectory

def rotate_traj(traj,angle,output_shape=(20,2)):
    """
    """

    angle_rad = np.deg2rad(angle)

    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c,-s], [s, c]])

    if traj.shape[1] != 2: # 2 x N -> N x 2
        trajectory = traj.transpose()
    else:
        trajectory = traj

    rotated_traj = np.matmul(trajectory,R) # (N x 2) x (2 x 2)
    # pdb.set_trace()
    if rotated_traj.shape[0] != output_shape[0]:
        try: # Numpy
            rotated_traj = rotated_traj.transpose()
        except: # Torch
            rotated_traj = torch.transpose(rotated_traj,0,1)

    return rotated_traj

# Goal points functions

NUM_GOAL_POINTS = 32

def get_points(img, car_px, scale_x, rad=100, color=255, N=1024, sample_car=True, max_samples=None):
    """
    """

    feasible_area = np.where(img == color)
    sampling_points = np.vstack(feasible_area)

    rng = default_rng()
    num_samples = N
    
    points_feasible_area = len(feasible_area[0])
    # num_samples = int(points_feasible_area/2)
    sample_index = rng.choice(points_feasible_area, size=N, replace=False)

    sampling_points = sampling_points[:,sample_index]
    
    px_y = sampling_points[0,:] # rows (pixels)
    px_x = sampling_points[1,:] # columns (pixels)
    
    ## sample points in the car radius
    
    if sample_car:
        final_points = [[a,b] for a,b in zip(px_y,px_x) if (math.sqrt(pow(a - car_px[0],2)+
                                                                      pow(b - car_px[1],2)) < rad)]

        if max_samples:               
            final_points = sample(final_points,max_samples)
            assert len(final_points) == max_samples
          
        final_points = np.array(final_points)

        try:
            px_y = final_points[:,0] # rows
            px_x = final_points[:,1] # columns
        except:
            scale_y = scale_x
            px_y = car_px[0] + scale_y*np.random.randn(num_samples) # columns
            px_x = car_px[1] + scale_x*np.random.randn(num_samples) # rows
                  
    return px_y, px_x

def change_bg_color(img):
    img_aux = copy.deepcopy(img)
    for i in range(img_aux.shape[0]):    
       for j in range(img_aux.shape[1]):  
           if (img_aux[i,j] == [0,0,0]).all():
               img_aux[i,j] = [255,255,255]
    return img_aux

# N.B. In PLT, points must be specified as standard cartesian frames (x from left to right, y from bottom to top)
def plot_fepoints(img, filename, obs_px_x, obs_px_y, car_px, 
                  goals_px_x=None, goals_px_y=None, radius=None, change_bg=False, show=False):
    assert len(img.shape) == 3
    
    img_aux = copy.deepcopy(img)
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.scatter(obs_px_x, obs_px_y, c="b", s=10) # Past trajectory
    plt.scatter(car_px[0], car_px[1], c="b", marker="*", s=50) # Last observation point
    if goals_px_x is not None:
        plt.scatter(goals_px_x, goals_px_y, color="purple", marker="x", s=10) # Goal points

    if change_bg:
        img_aux = change_bg_color(img)

    plt.imshow(img_aux)

    if radius:
      circ_car = plt.Circle((car_px[0], car_px[1]), radius, color="purple", fill=False)
      ax.add_patch(circ_car)

    plt.axis("off")

    if show:
        plt.title(filename) 
        plt.show()

    plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), 
                edgecolor='none', pad_inches=0)

    plt.close('all')

def get_agent_velocity(obs_seq, num_obs=5, period=0.1):
    """
    Consider the last num_obs points to calculate an average velocity of
    the object in the last observation point

    TODO: Consider an acceleration?
    """

    obs_seq_vel = obs_seq[:,-5:]

    vel = np.zeros((num_obs-1))

    for i in range(1,obs_seq_vel.shape[1]):
        x_pre, y_pre = obs_seq_vel[:,i-1]
        x_curr, y_curr = obs_seq_vel[:,i]

        dist = math.sqrt(pow(x_curr-x_pre,2)+pow(y_curr-y_pre,2))

        curr_vel = dist / period
        vel[i-1] = curr_vel

    return vel.mean()

def get_agent_yaw(obs_seq, num_obs=5):
    """
    Consider the last num_obs points to calculate an average yaw of
    the object in the last observation point
    """

    obs_seq_yaw = obs_seq[:,-5:]

    yaw = np.zeros((num_obs-1))

    for i in range(1,obs_seq_yaw.shape[1]):
        x_pre, y_pre = obs_seq_yaw[:,i-1]
        x_curr, y_curr = obs_seq_yaw[:,i]

        delta_y = y_curr - y_pre
        delta_x = x_curr - x_pre
        curr_yaw = math.atan2(delta_y, delta_x)

        yaw[i-1] = curr_yaw

    yaw = yaw[np.where(yaw != 0)]
    if len(yaw) == 0: # All angles were 0
        yaw = np.zeros((1))

    tolerance = 0.1
    num_positives = len(np.where(yaw > 0)[0])
    num_negatives = len(yaw) - num_positives
    final_yaw = yaw.mean()

    if (yaw.std() > 1.5): # and ((yaw.mean() > math.pi * (1 - tolerance) and yaw.mean() < - math.pi * (1 - tolerance)) # Around pi
                         #  or (yaw.mean() > -math.pi/12 * (1 - tolerance) and yaw.mean() < math.pi/12 * (1 - tolerance)))): # Around 0
        yaw = np.absolute(yaw)

        if num_negatives > num_positives:
            final_yaw = -yaw.mean()
        else:
            final_yaw = yaw.mean()

    return final_yaw

def transform_px2real_world(px_points, origin_pos, real_world_offset, img_size):
    """
    It is assumed squared image (e.g. 600 x 600 -> img_size = 600) and the same offset 
    in all directions (top, bottom, left, right) to facilitate the transformation.
    """

    xcenter, ycenter = origin_pos[0], origin_pos[1]
    x_min = xcenter - real_world_offset
    x_max = xcenter + real_world_offset
    y_min = ycenter - real_world_offset
    y_max = ycenter + real_world_offset

    m_x = float((2 * real_world_offset) / img_size) # slope
    m_y = float(-(2 * real_world_offset) / img_size) # slope

    i_x = x_min # intersection
    i_y = y_max

    rw_points = []

    for px_point in px_points:
        x = m_x * px_point[1] + i_x # Get x-real_world from columns
        y = m_y * px_point[0] + i_y # Get y-real_world from rows
        rw_point = [x,y]
        rw_points.append(rw_point)

    return np.array(rw_points)

def transform_real_world2px(rw_points, origin_pos, real_world_offset, img_size):
    """
    It is assumed squared image (e.g. 600 x 600 -> img_size = 600) and the same offset 
    in all directions (top, bottom, left, right) to facilitate the transformation.
    """

    xcenter, ycenter = origin_pos[0], origin_pos[1]
    x_min = xcenter - real_world_offset
    x_max = xcenter + real_world_offset
    y_min = ycenter - real_world_offset
    y_max = ycenter + real_world_offset

    m_x = float(img_size / (2 * real_world_offset)) # slope
    m_y = float(-img_size / (2 * real_world_offset))

    i_x = float(-(img_size / (2 * real_world_offset)) * x_min) # intercept
    i_y = float((img_size / (2 * real_world_offset)) * y_max)

    px_points = []

    for rw_point in rw_points:
        x = m_x * rw_point[0] + i_x
        y = m_y * rw_point[1] + i_y
        px_point = [x,y] 
        px_points.append(px_point)

    return np.array(px_points)

def get_goal_points(filename, obs_seq, origin_pos, real_world_offset):
    """
    """

    split_folder = '/'.join(filename.split('/')[:-2])
    goal_points_folder = split_folder + "/goal_points"
    seq_id = filename.split('/')[-1].split('.')[0]

    # 0. Load image and get past observations

    img = cv2.imread(filename)
    img = cv2.resize(img, dsize=(600,600))
    height, width = img.shape[:2]
    img_size = height
    scale_x = scale_y = float(height/(2*real_world_offset))

    cx = int(width/2)
    cy = int(height/2)
    car_px = (cy,cx)

    ori = obs_seq[-1, :]

    # 0. Plot obs traj (AGENT)

    obs_x = obs_seq[:,0]
    obs_y = obs_seq[:,1]

    obs_px_points = transform_real_world2px(obs_seq, origin_pos, real_world_offset, img_size)
    agent_obs_px_x, agent_obs_px_y = obs_px_points[:,0], obs_px_points[:,1]
    filename = goal_points_folder + "/" + seq_id + "_obs_traj.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, change_bg=True)

    # 1. Get feasible area points (N samples)

    # 1.0. (Optional) Observe random sampling in the whole feasible area

    fe_y, fe_x = get_points(img, car_px, scale_x, rad=10000, color=255, N=1024, 
                            sample_car=True, max_samples=None) # return rows, columns
    filename = goal_points_folder + "/" + seq_id + "_all_samples.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, goals_px_x=fe_x, goals_px_y=fe_y, change_bg=True)

    # 1.1. Filter using AGENT estimated velocity

    mean_vel = get_agent_velocity(torch.transpose(obs_seq,0,1))
    pred_seconds = 3 # instead of 3 s (prediction in ARGOVERSE)
    radius = mean_vel * pred_seconds
    radius_px = radius * scale_x	

    fe_y, fe_x = get_points(img, car_px, scale_x, rad=radius_px, color=255, N=1024, 
                                sample_car=True, max_samples=None) # return rows, columns

    filename = goal_points_folder + "/" + seq_id + "_vel_filter.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
    #               goals_px_x=fe_x, goals_px_y=fe_y, radius=radius_px, change_bg=True)

    # pdb.set_trace()

    # 1.2. Filter points applying rotation

    mean_yaw = get_agent_yaw(torch.transpose(obs_seq,0,1)) # radians

    if mean_yaw >= 0.0:
        angle = math.pi/2 - mean_yaw
    elif mean_yaw < 0.0:
        angle = -(math.pi / 2 + (math.pi - abs(mean_yaw)))

    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c,-s], [s, c]])

    fe_x_trans = fe_x - cx # get px w.r.t. the center of the image to be rotated
    fe_y_trans = fe_y - cy

    close_pts = np.hstack((fe_x_trans.reshape(-1,1),fe_y_trans.reshape(-1,1)))
    close_pts_rotated = np.matmul(close_pts,R).astype(np.int32)

    fe_x_rot = close_pts_rotated[:,0] + cx
    fe_y_rot = close_pts_rotated[:,1] + cy

    filtered_fe_x = fe_x[np.where(fe_y_rot < cy)[0]]
    filtered_fe_y = fe_y[np.where(fe_y_rot < cy)[0]]

    filename = goal_points_folder + "/" + seq_id + "_angle_filter.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
    #               goals_px_x=filtered_fe_x, goals_px_y=filtered_fe_y, radius=radius_px, change_bg=True)

    # 2. Get furthest N samples (closest the the hypothetical radius)

    dist = []
    for i in range(len(filtered_fe_x)):
        d = math.sqrt(pow(filtered_fe_x[i] - car_px[0],2) + pow(filtered_fe_y[i] - car_px[1],2))
        dist.append(d)

    dist = np.array(dist)

    np.argsort(dist)
    furthest_indeces = np.argsort(dist)[-NUM_GOAL_POINTS:]
    furthest_indeces

    final_samples_x, final_samples_y = filtered_fe_x[furthest_indeces], filtered_fe_y[furthest_indeces]
    
    try:
        diff_points = NUM_GOAL_POINTS - len(final_samples_x)
        final_samples_x = np.hstack((final_samples_x, final_samples_x[0]+0.2 * np.random.randn(diff_points)))
        final_samples_y = np.hstack((final_samples_y, final_samples_y[0]+0.2 * np.random.randn(diff_points)))
    except:
        final_samples_x = cx + scale_x*np.random.randn(NUM_GOAL_POINTS)
        final_samples_y = cy + scale_y*np.random.randn(NUM_GOAL_POINTS)

    filename = goal_points_folder + "/" + seq_id + "_final_samples.png"
    # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
    #               goals_px_x=final_samples_x, goals_px_y=final_samples_y, radius=radius_px, change_bg=True)

    if len(final_samples_x) != NUM_GOAL_POINTS:
        print(f"Final samples does not match with {NUM_GOAL_POINTS} required samples")
        # plot_fepoints(img, filename, agent_obs_px_x, agent_obs_px_y, car_px, 
        #               goals_px_x=final_samples_x, goals_px_y=final_samples_y, radius=radius_px, change_bg=True, show=True)
        pdb.set_trace()

    # 3. Transform pixels to real-world coordinates

    final_samples_px = np.hstack((final_samples_y.reshape(-1,1), final_samples_x.reshape(-1,1))) # rows, columns
    rw_points = transform_px2real_world(final_samples_px, origin_pos, real_world_offset, img_size)
    # pdb.set_trace()
    return rw_points