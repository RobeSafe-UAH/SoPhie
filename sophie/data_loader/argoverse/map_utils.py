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

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.mpl_plotting_utils import plot_bbox_2D
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

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

def viz_sequence(
    df: pd.DataFrame,
    lane_centerlines: Optional[List[np.ndarray]] = None,
    show: bool = True,
    smoothen: bool = False,
) -> None:

    # Seq data
    city_name = df["CITY_NAME"].values[0]

    if lane_centerlines is None:
        # Get API for Argo Dataset map
        avm = ArgoverseMap()
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    plt.figure(0, figsize=(8, 7), facecolor="black")

    x_min = min(df["X"]) 
    x_max = max(df["X"])
    y_min = min(df["Y"])
    y_max = max(df["Y"])

    if lane_centerlines is None:

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline

            if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        plt.plot(
            lane_cl[:, 0],
            lane_cl[:, 1],
            "-",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )
    frames = df.groupby("TRACK_ID")

    plt.xlabel("Map X")
    plt.ylabel("Map Y")

    color_dict = {"AGENT": "#d33e4c", "OTHERS": "#59dd4c", "AV": "#007672"}
    object_type_tracker: Dict[int, int] = defaultdict(int)

    # Plot all the tracks up till current frame
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        plt.plot(
            cor_x,
            cor_y,
            "-",
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            linewidth=1,
            zorder=_ZORDER[object_type],
        )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        # marker_type = "o"

        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 7
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 7
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 7

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

    red_star = mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=7, label="Agent")
    green_circle = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=7,
        label="Others",
    )
    black_triangle = mlines.Line2D([], [], color="black", marker="^", linestyle="None", markersize=7, label="AV")

    plt.axis("off")
    if show:
        plt.show()

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

    # relevant_lane_polygons = []
    # point = Point(origin[0][0],origin[0][1])
    # for pol in lane_polygons:
    #     my_pol = Polygon(pol)
    #     if point.within(my_pol):
    #         relevant_lane_polygons.append(pol)

    for i, polygon in enumerate(lane_polygons):
        if fill:
            ax.fill(polygon[:, 0], polygon[:, 1], "black", edgecolor='w', fill=True)
            # ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1.0, alpha=1.0, zorder=1)
        else:
            ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth, alpha=1.0, zorder=1)
            # ax.fill(polygon[:, 0], polygon[:, 1], edgecolor='b', fill=True)
        
def optimized_draw_lane_polygons(
    img: np.array,
    lane_polygons: np.ndarray,
    color: tuple,
) -> None:
    """Draw a lane using polygons.

    Args:
        img: Numpy array (image) where lanes must be painted
        lane_polygons: Array of (N,) objects, where each object is a (M,3) array
    """
    img_aux = copy.deepcopy(img)
    for i, polygon in enumerate(lane_polygons):
        cv_polygon = polygon[:,:2].reshape(-1,1,2).astype(np.int32)
        isClosed = False
        thickness = 2
        img_aux = cv2.polylines(img_aux, [cv_polygon], isClosed, color, thickness)

    return img_aux
    # print("img: ", img.shape, img)
    # if np.any(img):
    #     print("Any element different from zero")
    # assert 1 == 0

def rotate_polygon_about_pt(pts: np.ndarray, rotmat: np.ndarray, center_pt: np.ndarray) -> np.ndarray:
    """Rotate a polygon about a point with a given rotation matrix.

    Args:
        pts: Array of shape (N, 3) representing a polygon or point cloud
        rotmat: Array of shape (3, 3) representing a rotation matrix
        center_pt: Array of shape (3,) representing point about which we rotate the polygon

    Returns:
        rot_pts: Array of shape (N, 3) representing a ROTATED polygon or point cloud
    """
    pts -= center_pt
    rot_pts = pts.dot(rotmat.T)
    rot_pts += center_pt
    return rot_pts

def render_bev_labels_mpl(
        origin: np.array,
        city_name: str,
        ax: plt.Axes,
        axis: str,
        local_lane_polygons: np.ndarray,
        local_das: np.ndarray,
        city_to_egovehicle_se3: SE3,
        avm: ArgoverseMap,
    ) -> None:
        """Plot nearby lane polygons and nearby driveable areas (da) on the Matplotlib axes.

        Args:
            city_name: The name of a city, e.g. `"PIT"`
            img: Numpy array (image) where lanes must be painted
            axis: string, either 'ego_axis' or 'city_axis' to demonstrate the
            lidar_pts:  Numpy array of shape (N,3)
            local_lane_polygons: Polygons representing the local lane set
            local_das: Numpy array of objects of shape (N,) where each object is of shape (M,3)
            city_to_egovehicle_se3: Transformation from egovehicle frame to city frame
            avm: ArgoverseMap instance
        """
        if axis is not "city_axis":
            # rendering instead in the egovehicle reference frame
            for da_idx, local_da in enumerate(local_das):
                local_da = city_to_egovehicle_se3.inverse_transform_point_cloud(local_da)
                local_das[da_idx] = rotate_polygon_about_pt(local_da, city_to_egovehicle_se3.rotation, np.zeros(3))

            for lane_idx, local_lane_polygon in enumerate(local_lane_polygons):
                local_lane_polygon = city_to_egovehicle_se3.inverse_transform_point_cloud(local_lane_polygon)
                local_lane_polygons[lane_idx] = rotate_polygon_about_pt(
                    local_lane_polygon, city_to_egovehicle_se3.rotation, np.zeros(3)
                )

        draw_lane_polygons(ax, local_lane_polygons, "tab:blue", fill=False)
        draw_lane_polygons(ax, local_das, "tab:pink", fill=True)

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

    filled_img = np.zeros((*gray.shape,3))

    for cnt in contours:
        area = cv2.contourArea(cnt)
    
        # Shortlisting the regions based on there area.
        if area > 400: 
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        cv2.fillPoly(filled_img, pts=[cnt], color=(255, 255, 255))

    return filled_img

def optimized_render_bev_labels_mpl(
        origin_pos,
        img: np.array,
        local_lane_polygons: np.ndarray,
        local_das: np.ndarray,
    ) -> None:
        """Plot nearby lane polygons and nearby driveable areas (da) on the Matplotlib axes.

        Args:
            img: Numpy array (image) where lanes must be painted
            local_lane_polygons: Polygons representing the local lane set
            local_das: Numpy array of objects of shape (N,) where each object is of shape (M,3)
        """

        offset = np.array([[img.shape[0]/2,img.shape[1]/2]])
        for lp in local_lane_polygons:
            lp[:,:2] -= origin_pos
            lp[:,:2] += offset

        for lda in local_das:
            lda[:,:2] -= origin_pos
            lda[:,:2] += offset

        img_aux = optimized_draw_lane_polygons(img, local_lane_polygons, (255, 0, 0))
        img_aux = optimized_draw_lane_polygons(img_aux, local_das, (0, 255, 0))

        return img_aux

# Main function for map generation

def map_generator(seq: np.array, # Past_Observations · Num_agents x 2 (e.g. 200 x 2)
                  origin_pos,
                  offset,
                  avm,
                  city_name,
                  info, # object list id + num_agents_per_obs
                  show: bool = True,
                  smoothen: bool = False) -> None:

    plot_local_lane_polygons = True
    plot_local_das = True
    plot_centerlines = True
    plot_object_trajectories = True
    plot_object_heads = True

    xcenter, ycenter = origin_pos[0][0], origin_pos[0][1]
    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    object_id_list, obs = info

    # Get centerlines from Argoverse Map-API 

    t0 = time.time()
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    # Get local polygons

    local_lane_polygons = avm.find_local_lane_polygons([x_min, 
                                                        x_max, 
                                                        y_min, 
                                                        y_max], 
                                                        city_name)
    if not plot_local_lane_polygons:
        local_lane_polygons = []

    # Get driveable area from Argoverse Map-API 
    
    local_das = avm.find_local_driveable_areas([x_min, 
                                                x_max, 
                                                y_min, 
                                                y_max], 
                                                city_name)
    if not plot_local_das:
        local_das = []

    print("\nTime consumed by local das and polygons calculation: ", time.time()-t0)

    rotation = np.array([0,0,0,1]) # Quaternion with no roation
    translation = np.array([0,0,0]) # zero translation
    city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(rotation), translation=translation) # Just as argument, it is not used!

    t0 = time.time()
    fig = plt.figure(0, figsize=(6,6), facecolor="black")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ax = fig.add_subplot(111)
    # ax.set_facecolor((0.0, 0.0, 0.0)) # Black

    # render_bev_labels_mpl(
    #     origin_pos,
    #     city_name,
    #     ax,
    #     "city_axis",
    #     copy.deepcopy(local_lane_polygons),
    #     copy.deepcopy(local_das),
    #     city_to_egovehicle_se3,
    #     avm,
    # )
    draw_lane_polygons(ax, local_das, "tab:pink", linewidth=1.5, fill=True)

    img_cv = renderize_image(fig,normalize=False)
    filled_img = fill_driveable_area(img_cv)

    fig2 = plt.figure(0, figsize=(6,6), facecolor="black")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ax = fig2.add_subplot(111)

    draw_lane_polygons(ax, local_lane_polygons, "tab:red", linewidth=1.5, fill=False)

    print("Time consumed by BEV rendering: ", time.time()-t0)

    # Get lane centerlines which lie within the range of trajectories

    t0 = time.time()
    lane_centerlines = []
    # print("xmax,ymax,xmin,ymin: ", x_max,y_max,x_min,y_min)
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline

        if (np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min):
            lane_centerlines.append(lane_cl)

    if plot_centerlines:
        for lane_cl in lane_centerlines:

            plt.plot(
                lane_cl[:, 0],
                lane_cl[:, 1],
                "-",
                color="grey",
                alpha=1,
                linewidth=1.5,
                zorder=0,
            )

    print("Time consumed by plot lane centerlines: ", time.time()-t0)

    t0 = time.time()

    color_dict = {"AGENT": (0.0,0.0,1.0,1.0), # Blue (Red when represented in the image)
                  "AV": (1.0,0.0,0.0,1.0), # Red (Blue when represented in the image)
                  "OTHERS": (0.0,1.0,0.0,1.0)} 
    object_type_tracker: Dict[int, int] = defaultdict(int)

    obs_seq = seq[:200, :] # 200x2
    obs_seq_list = []
    for i in range(object_id_list.shape[0]):
        if object_id_list[i] != -1:
            obs_seq_list.append([obs_seq[np.arange(i,200,obs),:], object_id_list[i]]) # recover trajectories for each obs

    # Plot all the tracks up till current frame

    for seq_id in obs_seq_list:
        object_type = int(seq_id[1])
        seq_rel = seq_id[0]

        object_type = translate_object_type(object_type)

        cor_x = seq_rel[:,0] + xcenter
        cor_y = seq_rel[:,1] + ycenter

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]
        
        #pdb.set_trace()

        if plot_object_trajectories:
            plt.plot(
                cor_x,
                cor_y,
                "-",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                linewidth=3.0,
                zorder=_ZORDER[object_type],
            )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 10
        elif object_type == "OTHERS":
            marker_type = "*"
            marker_size = 8
        elif object_type == "AV":
            marker_type = "*"
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
    plt.axis("off")
    # plt.show()

    # Merge local driveable information and lanes information

    ## Foreground

    img_lanes = renderize_image(fig2,normalize=False)
    img2gray = cv2.cvtColor(img_lanes,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY)

    img2_fg = cv2.bitwise_and(img_lanes,img_lanes,mask=mask)
    cv2.imshow("foreground",img2_fg)

    ## Background

    filled_img = np.asarray(filled_img, np.uint8)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(filled_img,filled_img,mask=mask_inv)
    cv2.imshow("background",filled_img)

    ## Merge

    full_img = cv2.add(img1_bg,img2_fg)
    
    cv2.imshow("full_img",full_img)

    if show:
        plt.show()
    pdb.set_trace()
    return fig, fig2, full_img

def optimized_map_generator(
    seq: np.array, # Past_Observations · Num_agents x 2 (e.g. 200 x 2)
    origin_pos,
    offset,
    avm,
    city_name,
    info, # object list id + num_agents_per_obs
    show: bool = True,
    smoothen: bool = False,
) -> None:

    xcenter, ycenter = origin_pos[0][0], origin_pos[0][1]

    x_min = xcenter + offset[0]
    x_max = xcenter + offset[1]
    y_min = ycenter + offset[2]
    y_max = ycenter + offset[3]

    object_id_list, obs = info

    # Get centerlines from Argoverse Map-API 

    seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    # Get local polygons

    local_lane_polygons = avm.find_local_lane_polygons([x_min, 
                                                        x_max, 
                                                        y_min, 
                                                        y_max], 
                                                        city_name)
    
    # local_lane_polygons = []

    # Get driveable area from Argoverse Map-API 
    
    local_das = avm.find_local_driveable_areas([x_min, 
                                                x_max, 
                                                y_min, 
                                                y_max], 
                                                city_name)
    local_das = []

    rotation = np.array([0,0,0,1]) # Quaternion with no roation
    translation = np.array([0,0,0]) # zero translation
    city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(rotation), translation=translation) # Just as argument, it is not used!

    height = abs(offset[0])
    width = abs(offset[0])
    img = np.zeros((height,width,3))#, np.uint8)

    img_aux = optimized_render_bev_labels_mpl(origin_pos,
                                              img,
                                              copy.deepcopy(local_lane_polygons),
                                              copy.deepcopy(local_das),
                                              )

    curr_folder = os.getcwd()
    filename = curr_folder + "/hdmap_images/test_image_opencv_3.png"
    print("filename: ")
    cv2.imwrite(filename,img_aux)

    pdb.set_trace()

    lane_centerlines = []
    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in seq_lane_props.items():

        lane_cl = lane_props.centerline

        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        plt.plot(
            lane_cl[:, 0],
            lane_cl[:, 1],
            "-",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )

    t0 = time.time()

    color_dict = {"AGENT": (0.0,0.0,1.0,1.0), # Blue (Red when represented in the image)
                  "AV": (1.0,0.0,0.0,1.0), # Red (Blue when represented in the image)
                  "OTHERS": (1.0,1.0,1.0,1.0)} # White
    object_type_tracker: Dict[int, int] = defaultdict(int)

    seq_end = int(len(object_id_list)*20) # TODO manage magic number
    obs_seq = seq[:seq_end, :] # 200x2
    obs_seq_list = []
    # for i in range(object_id_list.shape[0]):
    #     if object_id_list[i] != -1:
    #         obs_seq_list.append([obs_seq[np.arange(i,200,obs),:], object_id_list[i]]) # recover trajectories for each obs

    try:
        for i in range(object_id_list.shape[0]):
            if object_id_list[i] != -1:
                obs_seq_list.append([obs_seq[np.arange(i,seq_end,obs),:], object_id_list[i]]) # recover trajectories for each obs
    except Exception as e:
        print(e)
        pdb.set_trace()
    # Plot all the tracks up till current frame

    for seq_id in obs_seq_list:
        object_type = int(seq_id[1])
        seq_rel = seq_id[0]

        object_type = translate_object_type(object_type)

        cor_x = seq_rel[:,0] + xcenter
        cor_y = seq_rel[:,1] + ycenter

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]
        
        #pdb.set_trace()

        plt.plot(
            cor_x,
            cor_y,
            "-",
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            linewidth=1,
            zorder=_ZORDER[object_type],
        )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 7
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 7
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 7

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

    plt.show()

    pdb.set_trace()

    plt.axis("off")
    if show:
        plt.show()
    return fig
