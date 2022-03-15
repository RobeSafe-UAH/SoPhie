import numpy as np
import pdb
import cv2
from numpy.random import default_rng
import math
import random
import pdb
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
from random import sample

FINAL_SAMPLES = 16

def get_points(img, car_px, rad=100, color=255, N=1024, sample_car=True, max_samples=None):
    """
    """

    feasible_area = np.where(img == color)
    
    rng = default_rng()
    num_samples = N
    
    points_feasible_area = len(feasible_area[0])
    sample_index = rng.choice(points_feasible_area, size=N, replace=False)

    sampling_points = np.vstack(feasible_area)
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
        px_y = final_points[:,0]
        px_x = final_points[:,1]
        
    return px_y, px_x

# N.B. In PLT, points must be specified as standard cartesian frames (x from left to right, y from bottom to top)
def plot_fepoints (img, px_x, px_y, car, radius=100):
    assert len(img.shape) == 3
    
    fig, ax = plt.subplots(figsize=(8, 6))

    
    #plt.scatter(x, y, c='r', s=10)
    plt.scatter(px_x, px_y, c='r', s=10)
    plt.scatter(car[0], car[1], c="blue", s=50)
    plt.imshow(img)
    circ_car = plt.Circle((car[0], car[1]), radius, color='b', fill=False)
    ax.add_patch(circ_car)
    plt.show()

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

  return yaw.mean()

img = cv2.imread("/home/robesafe/shared_home/img_map.png")
img = cv2.resize(img, dsize=(600,600))

agent_seq = np.array([[1766.14768904,  383.71936633], [1766.62052466,  384.10775593],
                    [1767.28550846,  384.6552796 ],
                    [1767.72575955,  385.00952349],
                    [1768.14065781,  385.46435397],
                    [1768.59842895,  385.82821033],
                    [1769.10951512,  386.24132036],
                    [1769.61903922,  386.67653916],
                    [1770.08811419,  387.11303431],
                    [1770.51685148,  387.45433246],
                    [1771.01198295,  387.85021631],
                    [1771.42921302,  388.17058808],
                    [1771.71824873,  388.39319543],
                    [1772.09306547,  388.7302689 ],
                    [1772.49395196,  389.05093178],
                    [1772.88982321,  389.38834941],
                    [1773.20873025,  389.64107915],
                    [1773.70903739,  390.17655078],
                    [1773.91188468,  390.26623965],
                    [1774.1851173 ,  390.52426648],
                    [1774.40452427,  390.67070236],
                    [1774.70613923,  390.94628352],
                    [1774.86931198,  391.10392481],
                    [1775.19464074,  391.46320853],
                    [1775.35392928,  391.49889976],
                    [1775.39334001,  391.59541151],
                    [1775.81005015,  391.88268262],
                    [1775.61648861,  391.89725624],
                    [1775.83474123,  392.03747647],
                    [1775.96701691,  392.16459204],
                    [1776.12807071,  392.31623024],
                    [1776.28836084,  392.45402221],
                    [1776.3916783 ,  392.53569916],
                    [1776.43712475,  392.60749732],
                    [1776.52714971,  392.65867589],
                    [1776.66808918,  392.79270137],
                    [1776.72960505,  392.85377694],
                    [1776.76756186,  392.86455324],
                    [1776.84420462,  392.95487347],
                    [1776.83743663,  392.94318012],
                    [1776.86021182,  392.98451847],
                    [1776.8971423 ,  393.03580808],
                    [1776.87214132,  393.01225724],
                    [1776.8529617 ,  392.95572977],
                    [1776.84682066,  392.98891709],
                    [1776.89000906,  393.01490595],
                    [1776.83048907,  392.9709569 ],
                    [1776.77142314,  392.94643682],
                    [1776.62709512,  392.69130151],
                    [1776.61946888,  392.68091922]])
obs_len = 20
obs_seq = agent_seq[:obs_len,:] # obs_len x 2 (x|y)

height,width = img.shape[:2]
cx = int(width/2)
cy = int(height/2)

real_world_offset = 40 # m
scale_x = scale_y = float(height/(2*real_world_offset))

car_px = (cy,cx)

# All feasible area points (N samples)

radius_px = 10000
fe_y, fe_x = get_points(img, car_px, rad=radius_px, color=255, N=256, sample_car=True, max_samples=None) # return rows, columns
plot_fepoints(img, fe_x, fe_y, car_px, radius=radius_px)

# Filter points using velocity

mean_vel = get_agent_velocity(obs_seq.transpose())
print("mean vel: ", mean_vel)

pred_seconds = 4 # instead of 3 s (prediction in ARGOVERSE)
radius = mean_vel * pred_seconds
radius_px = radius * scale_x

fe_y, fe_x = get_points(img, car_px, rad=radius_px, color=255, N=256, sample_car=True, max_samples=None) # return rows, columns
plot_fepoints(img, fe_x, fe_y, car_px, radius=radius_px)

# Filter points applying rotation

mean_yaw = get_agent_yaw(obs_seq.transpose()) # radians
print("mean yaw: ", mean_yaw)

c, s = np.cos(mean_yaw), np.sin(mean_yaw)
R = np.array([[c,-s], [s, c]])

fe_x_trans = fe_x - cx # get px w.r.t. the center of the image to be rotated
fe_y_trans = fe_y - cy

close_pts = np.hstack((fe_x_trans.reshape(-1,1),fe_y_trans.reshape(-1,1)))
close_pts_rotated = np.matmul(close_pts,R).astype(np.int32)

fe_x_rot = close_pts_rotated[:,0] + cx
fe_y_rot = close_pts_rotated[:,1] + cy

# plot_fepoints(img, fe_x_rot, fe_y_rot, car_px, radius=radius_px)

filtered_fe_x = fe_x[np.where(fe_y_rot < cy)[0]]
filtered_fe_y = fe_y[np.where(fe_y_rot < cy)[0]]

plot_fepoints(img, filtered_fe_x, filtered_fe_y, car_px, radius=radius_px)

# Get furthest N samples (closest the the hypothetical radius)

dist = []
for i in range(len(filtered_fe_x)):
  d = math.sqrt(pow(filtered_fe_x[i] - car_px[0],2) +
                pow(filtered_fe_y[i] - car_px[1],2))
  dist.append(d)

dist = np.array(dist)

np.argsort(dist)
furthest_indeces = np.argsort(dist)[-FINAL_SAMPLES:]
furthest_indeces

final_samples_x, final_samples_y = filtered_fe_x[furthest_indeces], filtered_fe_y[furthest_indeces]
plot_fepoints(img, final_samples_x, final_samples_y, car_px, radius=radius_px)

# Test transforms

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

# From px to real_world

final_samples_px = np.hstack((final_samples_y.reshape(-1,1), final_samples_x.reshape(-1,1))) # rows, columns
origin_pos = agent_seq[obs_len-1,:]
img_size = height
real_world_offset = 40

rw_points = transform_px2real_world(final_samples_px, origin_pos, real_world_offset, img_size)
plt.scatter(rw_points[:,0], rw_points[:,1], c='r', s=10)

x_min = origin_pos[0] - real_world_offset
x_max = origin_pos[0] + real_world_offset
y_min = origin_pos[1] - real_world_offset
y_max = origin_pos[1] + real_world_offset

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()

# From real_world to px

rec_px_points = transform_real_world2px(rw_points, origin_pos, real_world_offset, img_size)
rec_x, rec_y = rec_px_points[:,0], rec_px_points[:,1]
plot_fepoints(img, rec_x, rec_y, car_px, radius=radius_px)






