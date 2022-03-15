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
def plot_fepoints (img, px_x, px_y, car, radius=None):
    assert len(img.shape) == 3
    
    fig, ax = plt.subplots(figsize=(8, 6))

    
    #plt.scatter(x, y, c='r', s=10)
    plt.scatter(px_x, px_y, c='r', s=10)
    plt.scatter(car[0], car[1], c="blue", s=50)
    plt.imshow(img)

    if radius:
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
  # print("vel: ", vel)
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

#img = cv2.imread("/home/robesafe/shared_home/val_img_map_1.png")
#img = cv2.imread("/home/robesafe/shared_home/val_img_map_2.png")
img = cv2.imread("/home/robesafe/shared_home/val_img_map_3.png")
# img = cv2.imread("/home/robesafe/shared_home/val_img_map_4.png")
# img = cv2.imread("/home/robesafe/shared_home/val_img_map_5.png")

img = cv2.resize(img, dsize=(600,600))

# agent_seq = np.array([[1766.14768904,  383.71936633], # 1
#                     [1766.62052466,  384.10775593],
#                     [1767.28550846,  384.6552796 ],
#                     [1767.72575955,  385.00952349],
#                     [1768.14065781,  385.46435397],
#                     [1768.59842895,  385.82821033],
#                     [1769.10951512,  386.24132036],
#                     [1769.61903922,  386.67653916],
#                     [1770.08811419,  387.11303431],
#                     [1770.51685148,  387.45433246],
#                     [1771.01198295,  387.85021631],
#                     [1771.42921302,  388.17058808],
#                     [1771.71824873,  388.39319543],
#                     [1772.09306547,  388.7302689 ],
#                     [1772.49395196,  389.05093178],
#                     [1772.88982321,  389.38834941],
#                     [1773.20873025,  389.64107915],
#                     [1773.70903739,  390.17655078],
#                     [1773.91188468,  390.26623965],
#                     [1774.1851173 ,  390.52426648],
#                     [1774.40452427,  390.67070236],
#                     [1774.70613923,  390.94628352],
#                     [1774.86931198,  391.10392481],
#                     [1775.19464074,  391.46320853],
#                     [1775.35392928,  391.49889976],
#                     [1775.39334001,  391.59541151],
#                     [1775.81005015,  391.88268262],
#                     [1775.61648861,  391.89725624],
#                     [1775.83474123,  392.03747647],
#                     [1775.96701691,  392.16459204],
#                     [1776.12807071,  392.31623024],
#                     [1776.28836084,  392.45402221],
#                     [1776.3916783 ,  392.53569916],
#                     [1776.43712475,  392.60749732],
#                     [1776.52714971,  392.65867589],
#                     [1776.66808918,  392.79270137],
#                     [1776.72960505,  392.85377694],
#                     [1776.76756186,  392.86455324],
#                     [1776.84420462,  392.95487347],
#                     [1776.83743663,  392.94318012],
#                     [1776.86021182,  392.98451847],
#                     [1776.8971423 ,  393.03580808],
#                     [1776.87214132,  393.01225724],
#                     [1776.8529617 ,  392.95572977],
#                     [1776.84682066,  392.98891709],
#                     [1776.89000906,  393.01490595],
#                     [1776.83048907,  392.9709569 ],
#                     [1776.77142314,  392.94643682],
#                     [1776.62709512,  392.69130151],
#                     [1776.61946888,  392.68091922]])

# agent_seq = np.array([[1756.52242801,  375.75141426], # 2
#        [1757.40814636,  376.47177013],
#        [1758.39374164,  377.22899301],
#        [1759.34907712,  378.05457195],
#        [1760.25272839,  378.79206879],
#        [1761.1953549 ,  379.59114687],
#        [1762.10710471,  380.35388669],
#        [1763.06740782,  381.1032227 ],
#        [1764.02154834,  381.88159128],
#        [1764.98703603,  382.72902552],
#        [1765.95067459,  383.54496421],
#        [1766.91976593,  384.35324902],
#        [1767.82628513,  385.05295188],
#        [1768.81971061,  385.89390685],
#        [1769.55476138,  386.48117549],
#        [1770.86379934,  387.51607381],
#        [1771.86975183,  388.33595084],
#        [1772.59342946,  388.90644155],
#        [1773.83174628,  389.91834254],
#        [1774.85807799,  390.71731536],
#        [1775.91337818,  391.55162334],
#        [1776.496131  ,  392.037332  ],
#        [1777.69470565,  392.92948635],
#        [1778.69265921,  393.74671259],
#        [1780.04987818,  394.85778257],
#        [1781.00139521,  395.64690977],
#        [1782.04031997,  396.4959453 ],
#        [1783.03017578,  397.30169556],
#        [1784.0521996 ,  398.08127872],
#        [1784.79301901,  398.65536076],
#        [1786.09353032,  399.69077058],
#        [1786.78153719,  400.22299822],
#        [1787.88374359,  401.08218765],
#        [1788.79729945,  401.79634163],
#        [1789.73691694,  402.53475812],
#        [1791.10165439,  403.628435  ],
#        [1792.1627002 ,  404.43856961],
#        [1792.84708893,  404.91577455],
#        [1793.78980848,  405.71697199],
#        [1794.97326296,  406.59102023],
#        [1796.00587379,  407.44916948],
#        [1796.98804649,  408.15642325],
#        [1797.97015765,  408.99385704],
#        [1798.98994471,  409.77603235],
#        [1799.99889958,  410.52707879],
#        [1801.0484507 ,  411.31277923],
#        [1801.94548863,  412.06803973],
#        [1803.05532079,  412.90439074],
#        [1804.05134499,  413.65321529],
#        [1805.10247487,  414.47636996]])

agent_seq = np.array([[2163.96429805,  790.98399972], # 3
       [2163.42933103,  791.59425152],
       [2162.3429475 ,  792.82418809],
       [2161.36912059,  794.04072272],
       [2160.93387851,  794.54790607],
       [2159.93287517,  795.71195442],
       [2158.92743554,  796.84081119],
       [2158.48831804,  797.37275309],
       [2157.4559521 ,  798.51256923],
       [2156.37403205,  799.7018822 ],
       [2155.80369902,  800.38450846],
       [2154.91226614,  801.56944316],
       [2153.85575901,  802.54638317],
       [2153.43783804,  803.19652004],
       [2152.29603285,  804.48815399],
       [2151.92419964,  804.8339079 ],
       [2150.858191  ,  805.99866392],
       [2150.22985246,  806.94073584],
       [2149.65992109,  807.45796594],
       [2148.59135333,  808.71907163],
       [2148.19219153,  809.18426668],
       [2147.6086544 ,  809.84889978],
       [2146.89325663,  810.68067678],
       [2145.94451304,  811.76414917],
       [2145.34352879,  812.4885778 ],
       [2144.75667268,  813.1566508 ],
       [2143.61211962,  814.45074161],
       [2143.01190011,  815.18573372],
       [2142.39617518,  815.85588648],
       [2141.30240673,  817.14410588],
       [2140.75762672,  817.80871548],
       [2140.074625  ,  818.52578138],
       [2138.89497752,  819.87677309],
       [2138.28599764,  820.5487331 ],
       [2137.08265798,  821.9655697 ],
       [2136.60406213,  822.5979905 ],
       [2136.11984338,  823.15956199],
       [2134.82085025,  824.67292877],
       [2134.37906552,  825.06607259],
       [2133.84355106,  825.67416136],
       [2133.30396036,  826.31147568],
       [2132.04101593,  827.97895232],
       [2131.55427934,  828.67755696],
       [2130.99080122,  829.31853142],
       [2130.06431851,  830.5202565 ],
       [2129.44343118,  831.21816674],
       [2128.72551349,  832.11176189],
       [2127.73099076,  833.29531878],
       [2127.12917073,  834.00645363],
       [2126.35335356,  834.90741155]])

agent_seq = np.array([[1740.9016416 ,  362.59457515], # 4
       [1742.06334923,  363.54933515],
       [1743.42555333,  364.69180193],
       [1745.92655294,  366.28358133],
       [1745.92655294,  366.28358133],
       [1746.89520447,  367.30412206],
       [1748.1661697 ,  368.26341519],
       [1749.37294109,  369.46040725],
       [1750.43931814,  370.41519606],
       [1751.67089741,  371.20117231],
       [1752.93318137,  372.24016283],
       [1755.27540715,  374.0284414 ],
       [1755.27540715,  374.0284414 ],
       [1756.19561545,  374.94270327],
       [1757.55379782,  376.00529829],
       [1758.62461357,  376.9732836 ],
       [1759.81479111,  377.80569332],
       [1760.78755862,  378.67325418],
       [1761.6093477 ,  379.39254618],
       [1762.57972427,  380.29431232],
       [1763.48055786,  381.15502821],
       [1764.946147  ,  382.31188017],
       [1765.86623144,  383.11229106],
       [1767.12921511,  384.09306857],
       [1768.18188236,  385.05035924],
       [1769.19517035,  385.89546741],
       [1770.31970234,  386.840454  ],
       [1771.46513136,  387.83598559],
       [1772.68306905,  388.79286781],
       [1773.71400818,  389.71734925],
       [1774.53391565,  390.43641961],
       [1777.80190837,  393.00245703],
       [1778.70433682,  393.78257222],
       [1779.71909129,  394.59640835],
       [1781.0964408 ,  395.95657555],
       [1782.04111811,  396.69586784],
       [1783.17968214,  397.68891   ],
       [1784.46388622,  398.78704307],
       [1785.47704625,  399.6694949 ],
       [1786.73290971,  400.68063422],
       [1787.42952054,  401.51105164],
       [1788.85328582,  402.53944693],
       [1790.10400909,  403.57364163],
       [1791.37962302,  404.63721474],
       [1792.2686886 ,  405.45257022],
       [1793.18507064,  406.19768438],
       [1794.32492379,  407.1046483 ],
       [1796.65365467,  409.34004786],
       [1797.88385309,  410.25092715],
       [1798.30236801,  410.49426751]])

# agent_seq = np.array([[ 850.0359866 , 1547.45075618], # 5
#        [ 850.36277309, 1547.99096749],
#        [ 850.59665025, 1548.42556464],
#        [ 850.92700306, 1548.95590476],
#        [ 851.38657219, 1549.71693876],
#        [ 851.60195093, 1550.08703444],
#        [ 851.91583261, 1550.62554032],
#        [ 852.96634849, 1552.42450192],
#        [ 853.64343989, 1553.61281836],
#        [ 854.70675023, 1555.39747389],
#        [ 855.7605304 , 1557.16692359],
#        [ 856.01595854, 1558.30613895],
#        [ 856.89511578, 1559.50582642],
#        [ 857.59317141, 1560.7472108 ],
#        [ 858.21674691, 1562.12683559],
#        [ 859.32799539, 1564.22383098],
#        [ 860.16934071, 1565.6011237 ],
#        [ 861.23843383, 1568.11857652],
#        [ 862.14286232, 1569.17383885],
#        [ 863.16055508, 1571.69136728],
#        [ 864.35620468, 1573.98256361],
#        [ 865.0626378 , 1575.48503265],
#        [ 865.98156797, 1576.78449737],
#        [ 866.81060475, 1578.40880249],
#        [ 867.62593048, 1580.1266963 ],
#        [ 868.59080009, 1582.76097884],
#        [ 869.17980279, 1583.39795586],
#        [ 869.82222467, 1585.05425152],
#        [ 870.6333553 , 1586.92537629],
#        [ 871.75106947, 1589.57142007],
#        [ 872.74706114, 1592.08185356],
#        [ 873.54948108, 1595.01841563],
#        [ 874.2745422 , 1596.93316693],
#        [ 874.82387819, 1598.83066555],
#        [ 875.75189053, 1601.53848778],
#        [ 876.29704593, 1603.41524227],
#        [ 876.8531112 , 1605.33809946],
#        [ 877.31596179, 1607.2033291 ],
#        [ 877.77790771, 1609.12294535],
#        [ 878.23816861, 1611.21926402],
#        [ 878.70063832, 1613.09130137],
#        [ 879.09100521, 1615.14081145],
#        [ 879.49916914, 1616.05233675],
#        [ 879.70690835, 1618.32817337],
#        [ 880.02413461, 1620.23032707],
#        [ 880.35360817, 1622.21378533],
#        [ 880.78216626, 1625.3303396 ],
#        [ 881.24135195, 1629.82992244],
#        [ 881.34632166, 1632.00014825],
#        [ 881.48032124, 1634.05020081]])

obs_len = 20
obs_seq = agent_seq[:obs_len,:] # obs_len x 2 (x|y)

height,width = img.shape[:2]

cx = int(width/2)
cy = int(height/2)
origin_pos = agent_seq[obs_len-1,:]

real_world_offset = 40 # m
img_size = height
scale_x = scale_y = float(height/(2*real_world_offset))

car_px = (cy,cx)

# Obs traj

obs_x = obs_seq[:,0]
obs_y = obs_seq[:,1]

obs_px_points = transform_real_world2px(obs_seq, origin_pos, real_world_offset, img_size)
rec_obs_x, rec_obs_y = obs_px_points[:,0], obs_px_points[:,1]
plot_fepoints(img, rec_obs_x, rec_obs_y, car_px)

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

angle = math.pi/2 - mean_yaw

c, s = np.cos(angle), np.sin(angle)
R = np.array([[c,-s], [s, c]])

fe_x_trans = fe_x - cx # get px w.r.t. the center of the image to be rotated
fe_y_trans = fe_y - cy

close_pts = np.hstack((fe_x_trans.reshape(-1,1),fe_y_trans.reshape(-1,1)))
close_pts_rotated = np.matmul(close_pts,R).astype(np.int32)

fe_x_rot = close_pts_rotated[:,0] + cx
fe_y_rot = close_pts_rotated[:,1] + cy

plot_fepoints(img, fe_x_rot, fe_y_rot, car_px, radius=radius_px)

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






