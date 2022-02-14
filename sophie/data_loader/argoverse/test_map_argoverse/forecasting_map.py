import time
import os
import numpy as np
import sys
import pdb

from argoverse.map_representation.map_api import ArgoverseMap
from matplotlib.pyplot import show

lib_path = os.path.join("/home", "robesafe", "libraries")
if os.path.exists(lib_path):
    sys.path.append("/home/robesafe/libraries/SoPhie")
import sophie.data_loader.argoverse.map_utils as map_utils
import cv2

city_name = None
avm = ArgoverseMap()

curr_obs_seq_data = np.load('relative_sequences.npy')
city_name = str(np.load("city_id.npy"))
curr_ego_origin = np.load("origin.npy")
obj_id_list = np.load("obj_id_list.npy")
num_agents_per_obs = 10

d = 40
dist_rasterized_map = [-d,d,-d,d]

# print("relative_seq ", relative_seq.shape)
# print("ego_origin ", ego_origin.shape)
# print("city_name: ", city_name, type(city_name))

t0 = time.time()
fig = map_utils.map_generator(curr_obs_seq_data, 
# fig = map_utils.optimized_map_generator(curr_obs_seq_data, 
                              curr_ego_origin, 
                              dist_rasterized_map, 
                              avm, 
                              city_name,
                              (obj_id_list, num_agents_per_obs), 
                              show=False, 
                              smoothen=True)

print("Time consumed by map generation: ", time.time() - t0)
t0 = time.time()
img_render = map_utils.renderize_image(fig)
img = img_render * 255.0
print("Time consumed by image rendering: ", time.time() - t0)

# Find contours

gray = cv2.cvtColor(img_render.astype(np.float32), cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)   
# pdb.set_trace()

_,threshold = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY)
threshold = threshold.astype(np.uint8)

# cv2.imshow('threshold', threshold) 
indices = np.where(threshold == 255)
threshold = threshold[mask]

pdb.set_trace()

contours,_=cv2.findContours(threshold, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
# print("Contours: ", contours)
# cv2.imshow('threshold', threshold) 
# pdb.set_trace()
# Searching through every region selected to 
# find the required polygon.
img2 = np.zeros((img.shape))

# print("cnt: ",contours, type(contours))
# pdb.set_trace()
# contours.reverse()
# contours = [np.vstack(contours).reshape(-1,1,2)]
# print("cnt2: ", contours)
cont = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    print("area: ", area)
   
    # Shortlisting the regions based on there area.
    # if area > 400: 
    #     approx = cv2.approxPolyDP(cnt, 
    #                               0.009 * cv2.arcLength(cnt, True), True)
   
    #     # Checking if the no. of sides of the selected region is 7.
    #     # if(len(approx) == 7): 
    #     # cv2.drawContours(img2, [approx], 0, (0, 0, 255), 2)
    cv2.drawContours(img2, [cnt], 0, (0, 0, 255),8)
    # pt1 = tuple(cnt[0][0])
    # pt2 = tuple(cnt[-1][0])

    # cv2.circle(img2,pt1,radius=5,color=(255,0,0),thickness=-1)
    # cv2.circle(img2,pt2,radius=5,color=(255,0,0),thickness=-1)
    cv2.fillPoly(img2, pts=[cnt], color=(255, 255, 255))
   
# Showing the image along with outlined arrow.
cv2.imshow('image2', img2)   
pdb.set_trace()

curr_folder = os.getcwd()
filename = curr_folder + "/hdmap_images/test_image_plt.png"
cv2.imwrite(filename,img)
