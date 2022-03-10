#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 7 12::33:19 2022
@author: Carlos Gómez-Huélamo and Miguel Eduardo Ortiz Huamaní
"""

from bdb import set_trace
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from shapely.geometry import Polygon, MultiPolygon, Point
import geopandas as gpd
import copy
import logging
import sys
import time
import pdb
import os

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp

from argoverse.utils.centerline_utils import (
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    lane_waypt_to_query_dist,
    remove_overlapping_lane_seq,
)

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.mpl_plotting_utils import plot_bbox_2D
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

from sophie.utils.utils import relative_to_abs
import sophie.data_loader.argoverse.dataset_utils as dataset_utils

IS_OCCLUDED_FLAG = 100
LANE_TANGENT_VECTOR_SCALING = 4
plot_lane_tangent_arrows = True

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Aux functions

def renderize_image(fig_plot, new_shape=(600,600),normalize=True):
    fig_plot.canvas.draw()

    img_cv2 = cv2.cvtColor(np.asarray(fig_plot.canvas.buffer_rgba()), cv2.COLOR_RGBA2RGB)
    # img_rsz = cv2.resize(img_cv2, new_shape)#.astype(np.float32)

    # gray = cv2.cvtColor(img_rsz.astype(np.float32), cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img_cv2.astype(np.float32), cv2.COLOR_BGR2GRAY)
    rows,columns = np.where(gray != 0)
    img_cv2 = img_cv2[rows.min():rows.max(),
                      columns.min():columns.max(),
                      :] # Remove padding
    img_rsz = cv2.resize(img_cv2, new_shape)#.astype(np.float32)                  
    
    if normalize:
        img_rsz = img_rsz / 255.0 # Normalize from 0 to 1
    return img_rsz

_ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}

def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def translate_object_type(int_id):
    if int_id == 0:
        return "AV"
    elif int_id == 1:
        return "AGENT"
    else:
        return "OTHERS"

def draw_lane_polygons(
    ax: plt.Axes,
    lane_polygons: np.ndarray,
    color: Union[Tuple[float, float, float], str],
    linewidth:float,
    fill: bool,
) -> None:
    """Draw a lane using polygons.

    Args:
        ax: Matplotlib axes
        lane_polygons: Array of (N,) objects, where each object is a (M,3) array
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
    """

    for i, polygon in enumerate(lane_polygons):
        if fill:
            ax.fill(polygon[:, 0], polygon[:, 1], "black", edgecolor='w', fill=True)
        else:
            ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth, alpha=1.0, zorder=1)

def fill_driveable_area(img_render):
    """
    """

    img = img_render * 255.0

    # Find limits of the driveable area

    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    rows,columns = np.where(gray != 0)
    gray = gray[rows.min():rows.max(),
                columns.min():columns.max()] # Remove padding
    gray = cv2.resize(gray, (600,600))#.astype(np.float32)  

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

    # Find contours

    _,threshold = cv2.threshold(gray, 0, 255, 
                                cv2.THRESH_BINARY)
    threshold = threshold.astype(np.uint8)
    contours,_= cv2.findContours(threshold, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    # Fill driveable area

    filled_img = 255 * np.ones((*gray.shape,3))

    for cnt in contours:
        area = cv2.contourArea(cnt)
    
        # Shortlisting the regions based on there area.
        if area > 400: 
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        # cv2.fillPoly(filled_img, pts=[cnt], color=(255, 255, 255))
        cv2.fillPoly(filled_img, pts=[cnt], color=(0, 0, 0))

    return filled_img

# Main function for map generation

def map_generator(curr_num_seq,
                  origin_pos,
                  offset,
                  avm,
                  city_name,
                  show: bool = True) -> None:
    """
    """

    plot_centerlines = True
    plot_local_das = True
    plot_local_lane_polygons = False

    xcenter, ycenter = origin_pos[0][0], origin_pos[0][1]
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    # Get map info 

    ## Get centerlines around the origin

    t0 = time.time()

    seq_lane_props = dict()
    if plot_centerlines:
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    ### Get lane centerlines which lie within the range of trajectories

    lane_centerlines = []
    
    for lane_id, lane_props in seq_lane_props.items():

        lane_cl = lane_props.centerline

        if (np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min):

            lane_centerlines.append(lane_cl)

    ## Get local polygons around the origin

    local_lane_polygons = []
    if plot_local_lane_polygons:
        local_lane_polygons = avm.find_local_lane_polygons([x_min, 
                                                            x_max, 
                                                            y_min, 
                                                            y_max], 
                                                            city_name)

    ## Get drivable area around the origin

    local_das = []
    if plot_local_das:
        local_das = avm.find_local_driveable_areas([x_min, 
                                                    x_max, 
                                                    y_min, 
                                                    y_max], 
                                                    city_name)

    # print("\nTime consumed by local das and polygons calculation: ", time.time()-t0)

    # Plot

    fig, ax = plt.subplots(figsize=(6,6), facecolor="black")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("off") # Uncomment if you want to generate images with x|y-labels
    
    ## Plot nearby segments

    # avm.plot_nearby_halluc_lanes(ax, city_name, xcenter, ycenter)

    ## Plot hallucinated polygones and centerlines

    t0 = time.time()

    for lane_cl in lane_centerlines:
        lane_polygon = centerline_to_polygon(lane_cl[:, :2])
                                                                          #"black"  
        ax.fill(lane_polygon[:, 0], lane_polygon[:, 1], "white", edgecolor='white', fill=True)
                                                        #"grey"
        ax.plot(lane_cl[:, 0], lane_cl[:, 1], "-", color="black", linewidth=1.5, alpha=1.0, zorder=1)

    # print("Time consumed by plot drivable area and lane centerlines: ", time.time()-t0)

    # draw_lane_polygons(ax, local_das, "tab:pink", linewidth=1.5, fill=False)
    # filled_img = fill_driveable_area(img_cv) # Not test for complex polygons

    # ax.clear()
    # ax.set_facecolor((0.0,0.0,0.0))

    # draw_lane_polygons(ax, local_lane_polygons, "tab:red", linewidth=1.5, fill=False)

    full_img_cv = renderize_image(fig,new_shape=(224,224),normalize=False)
    # full_img_cv = cv2.bitwise_not(full_img_cv)

    if show:
        cv2.imshow("full_img",full_img_cv)

    root_folder = "/home/robesafe/libraries/SoPhie/data/datasets/argoverse/motion-forecasting/train/data_images"
    filename = root_folder + "/" + str(curr_num_seq) + ".png"
    cv2.imwrite(filename,full_img_cv)

    return full_img_cv

def plot_trajectories(filename,obs_seq,first_obs,origin_pos, object_class_id_list,offset,\
                      rotation_angle=0,obs_len=None,smoothen=False,show=False):
    """
    Plot until plot_len points per trajectory. If plot_len != None, we
    must distinguish between observation (color with a marker) and prediction (same color with another marker)
    """

    xcenter, ycenter = origin_pos[0][0], origin_pos[0][1]
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    plot_object_trajectories = True
    plot_object_heads = True

    # img_map = plt.imread(filename)
    img_map = cv2.imread(filename)
    height,width,channels = img_map.shape

    fig, ax = plt.subplots(figsize=(6,6), facecolor="white")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("off") # Uncomment if you want to generate images with

    t0 = time.time()

    color_dict = {"AGENT": (0.0,0.0,1.0,1.0), # BGR
                  "AV": (1.0,0.0,0.0,1.0), 
                  "OTHERS": (0.37,0.37,0.37,1.0)} 
    object_type_tracker: Dict[int, int] = defaultdict(int)

    obs_seq_list = []
    
    for i in range(len(object_class_id_list)):
        obs_ = obs_seq[:,i,:].view(-1,2) # 20 x 2 (rel-rel)
        curr_first_obs = first_obs[i,:].view(-1)

        abs_obs_ = relative_to_abs(obs_, curr_first_obs) # "abs" (around 0)
        obj_id = object_class_id_list[i]
        obs_seq_list.append([abs_obs_,obj_id])

    # Plot all the tracks up till current frame

    for seq_id in obs_seq_list:
        object_type = int(seq_id[1])
        seq_rel = seq_id[0]

        object_type = translate_object_type(object_type)

        cor_x = seq_rel[:,0] + xcenter #+ width/2
        cor_y = seq_rel[:,1] + ycenter #+ height/2

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        if plot_object_trajectories:
            plt.plot(
                cor_x,
                cor_y,
                "-",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                linewidth=2.5,
                zorder=_ZORDER[object_type],
            )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 8
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 8
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 8

        if plot_object_heads:
            plt.plot(
                final_x,
                final_y,
                marker_type,
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                markersize=marker_size,
                zorder=_ZORDER[object_type],
            )

        object_type_tracker[object_type] += 1
    print("Time consumed by objects rendering: ", time.time()-t0)

    # Merge local driveable information and trajectories information

    ## Foreground

    img_lanes = renderize_image(fig,new_shape=(224,224),normalize=False)
    img2gray = cv2.cvtColor(img_lanes,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,127,255,cv2.THRESH_BINARY_INV)
    img2_fg = cv2.bitwise_and(img_lanes,img_lanes,mask=mask)

    ## Background

    mask_inv = cv2.bitwise_not(mask)
    # img_map = cv2.rotate(img_map, cv2.ROTATE_90_CLOCKWISE)#ROTATE_180)
    img1_bg = cv2.bitwise_and(img_map,img_map,mask=mask_inv)

    ## Merge

    full_img_cv = cv2.add(img1_bg,img2_fg)

    if show:
        cv2.imshow("full_img",full_img_cv)

    norm_full_img_cv = full_img_cv / 255.0

    return norm_full_img_cv
