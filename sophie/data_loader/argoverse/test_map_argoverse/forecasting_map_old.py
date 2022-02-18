import time
import os
import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point

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

# Find limits of the driveable area

gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
rows,columns = np.where(gray != 0)
padding_size = 0
gray = gray[rows.min()-padding_size:rows.max()+padding_size,
            columns.min()-padding_size:columns.max()+padding_size] # Remove padding
# rows,columns = np.where(gray != 0)
_,gray = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY)
rows,columns = np.where(gray != 0)
thickness = 1
row_top_edge_points = np.where(rows == 0)[0]
min_,max_ = row_top_edge_points[0], row_top_edge_points[-1]
gray = cv2.line(gray,(columns[min_],0),(columns[max_],0),(255,255,255),thickness)

row_bottom_edge_points = np.where(rows == (gray.shape[0]-1))[0]
min_,max_ = row_bottom_edge_points[0], row_bottom_edge_points[-1]
gray = cv2.line(gray,(columns[min_],gray.shape[0]-1),(columns[max_],gray.shape[0]-1),(255,255,255),thickness)

column_left_edge_points = np.where(columns == 0)[0]
min_,max_ = column_left_edge_points[0], column_left_edge_points[-1]
gray = cv2.line(gray,(0,rows[min_]),(0,rows[max_]),(255,255,255),thickness)

column_right_edge_points = np.where(columns == (gray.shape[1]-1))[0]
min_, max_ = column_right_edge_points[0], column_right_edge_points[-1]
gray = cv2.line(gray,(gray.shape[1]-1,rows[min_]),(gray.shape[1]-1,rows[max_]),(255,255,255),thickness)

padding_size = 2
value = [255, 255, 255]
new_gray = np.zeros((*gray.shape,3))
# new_gray = cv2.copyMakeBorder(new_gray, padding_size, padding_size, 
#                           padding_size, padding_size, cv2.BORDER_CONSTANT, value=value)
# gray = cv2.copyMakeBorder(gray, padding_size, padding_size, 
#                           padding_size, padding_size, cv2.BORDER_CONSTANT, value=value)
rows,columns = np.where(gray != 0)
unique_rows = np.unique(rows)
unique_columns = np.unique(columns)

cv2.imshow('prev gray', gray) 

# for row in unique_rows:
#     rows_indeces = np.where(rows==row)[0]
#     min_col_index, max_col_index = columns[rows_indeces[0]], columns[rows_indeces[-1]]
#     print("index, reps, min, max: ", row, len(rows_indeces), min_col_index, max_col_index)
#     diff = max_col_index - min_col_index
#     print("diff: ", diff)
#     print("----------")
#     gray[row,min_col_index:max_col_index] = 255
print("rows: ", len(rows))
print("columns: ", len(columns))
print("shape: ", gray.shape)


# _,gray = cv2.threshold(gray, 0, 255, 
#                             cv2.THRESH_BINARY)
# paint = False
# horz_line_cont = 0
# ranges_list = []
# num_errors = 0
# for row in range(gray.shape[0]):
#     # if (row == gray.shape[0]-1 or row == 0):
#     #     continue
#     cnt = 0
#     paint = False
#     range_list = []
#     horz_line_cont = 0
#     # print("row: ", row)
#     for column in range(gray.shape[1]-1):
#         # if (column == 0 or column == gray.shape[1]-1):
#         #     continue
#         # if paint:
#         #     new_gray[row][column] = 255
#         # # if row == 2: print("cnt: ", cnt)
#         # # print("value: ", gray[row][column])
#         if gray[row][column] == 255.0:
#             new_gray[row][column] = 255.0
#             horz_line_cont += 1
#         else:
#             horz_line_cont = 0
#         if ((gray[row][column] != 0.0 and gray[row][column+1] == 0.0) and horz_line_cont < 10):
#             # print("Sumo")
#             cnt += 1
#             range_list.append(column)
#         # if cnt % 2 != 0:
#         #     # print("pinto")
#         #     paint = True
#         # else:
#         #     paint = False
#     print("range_list: ", range_list)
#     try:
#         if len(range_list) > 1:
#             new_gray[row][range_list[0]:range_list[-1]] = 255.0
#     except:
#         num_errors += 1
#         pt1 = (int(gray.shape[1]/2),row)
#         cv2.circle(new_gray,pt1,radius=5,color=(255,0,0),thickness=-1)
#         # print("ERROR")
#     # if row == 2:
#     #     pdb.set_trace()

# print("num errors: ", num_errors)
# cv2.imshow('new gray', new_gray) 
# pdb.set_trace()

# Find contours

_,threshold = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY)
threshold = threshold.astype(np.uint8)
cv2.imshow('threshold', threshold) 
contours,_= cv2.findContours(threshold, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

print("Contours: ", contours)
# cv2.imshow('threshold', threshold) 
# pdb.set_trace()
# Searching through every region selected to 
# find the required polygon.

# print("cnt: ",contours, type(contours))
# pdb.set_trace()
# contours.reverse()
# contours = np.vstack(contours).reshape(-1,1,2)
# my_pol = Polygon(contours.reshape(contours.shape[0],2))

# plt.plot(*my_pol.exterior.xy)
# plt.show()
# p = gpd.GeoSeries(my_pol)
# p.plot()
# plt.show()

# pdb.set_trace()
# list_arrays = [ np.array((geom.xy[0][0], geom.xy[1][0])) for geom in my_pol ]

# for array in list_arrays:
#     print (array)
# pdb.set_trace()
# # contours = contours[::-1]
# contours = [contours]
# print("cnt2: ", contours)
# cont = 0


# for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#         point = Point(i,j)
#         if point.within(my_pol):
#             gray[i][j] = 255
# pdb.set_trace()
for cnt in contours:
    area = cv2.contourArea(cnt)
    print("area: ", area)
   
    # Shortlisting the regions based on there area.
    if area > 400: 
        approx = cv2.approxPolyDP(cnt, 
                                  0.009 * cv2.arcLength(cnt, True), True)
   
        # Checking if the no. of sides of the selected region is 7.
    #     if(len(approx) == 7): 
    #         cv2.drawContours(gray, [approx], 0, (0, 0, 255), 15)

    # font                   = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale              = 0.1
    # fontColor              = (255,255,255)
    # thickness              = 1
    # lineType               = 2
    # for i in range(60):
    #     pt = tuple(cnt[i][0])
    #     cv2.circle(gray,pt,radius=5,color=(255,255,255),thickness=-1)
    #     cv2.putText(gray,str(i),pt,font, fontScale,fontColor,thickness,lineType)
    # cv2.drawContours(new_gray, [cnt], 0, (0, 0, 255),5)
    cv2.fillPoly(new_gray, pts=[cnt], color=(255, 255, 255))
    # pt1 = tuple(cnt[0][0])
    # pt2 = tuple(cnt[-1][0])
cv2.imshow('image44', new_gray)   
pdb.set_trace()
    # cv2.circle(gray,pt1,radius=15,color=(255,0,0),thickness=-1)
    # cv2.circle(gray,pt2,radius=15,color=(255,0,0),thickness=-1)

# contours = contours[0][:10]

# # contours = [np.array([[50,50], [50,150], [150,150], [150,50]])]
# print("contours: ", contours, type(contours))

   
# Showing the image along with outlined arrow.
cv2.imshow('image44', gray)   
pdb.set_trace()

curr_folder = os.getcwd()
filename = curr_folder + "/hdmap_images/test_image_plt.png"
cv2.imwrite(filename,gray)
