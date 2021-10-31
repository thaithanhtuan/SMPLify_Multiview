"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

About this Script:
============
This is a demo version of the algorithm implemented in the paper,
which fits the SMPL body model to the image given the joint detections.
The code is organized to be run on the LSP dataset.
See README to see how to download images and the detected joints.


==============
Modification : Heejune

dependent 
===============

numpy    (pip install numpy) 
opencv   (pip install opencv-python)
chumpy   (pip install chumpy)
smpl     (modified by seoultech)
smplify  (named for package) 

"""

import json
import os
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
# import cPickle as pickle   # Python 2
import _pickle as pickle   # Python 3
from time import time
import open3d as o3d

"""
Note:
{
  "Color_intrinsics" : [1280.000000, 720.000000, 605.927002, 605.799927, 639.942627, 367.209930],
  "Depth_intrinsics" : [640.000000, 576.000000, 502.724945, 502.854401, 323.970764, 326.964050]
}
(width, height, fx, fy, cx, cy)

"""
import inspect  # for debugging

# basics
import cv2
import math
import numpy as np
import chumpy as ch
from scipy.spatial import cKDTree

from utils import *

# SMPL
from smpl.serialization import load_model
from smplify_core import run_single_fit

# Trimesh
import trimesh

import smpl_pyrender
import pyrender

from clothes import make_clothes


_LOGGER = logging.getLogger(__name__)

""" 

   Sinle Image Fitting Function  

"""
def remove_small_clusters(tri_pcd, eps = 5, min_points = 2):
    voxel_size = 20
    points =np.array(tri_pcd.vertices)
    colors = np.array(tri_pcd.colors)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    seglabels = pcd.cluster_dbscan(eps=voxel_size*eps, min_points=min_points, print_progress=True)
    # pcd.cluster_dbscan(eps=20, min_points=min_points, print_progress=True)
    labelset, label_counts = np.unique(seglabels, return_counts = True)
    if(np.any(labelset == -1) == True):
        labelset = labelset[1:]
        label_counts = label_counts[1:]
    max_idx = np.argmax(label_counts)
    foot_label = labelset[max_idx]
    colors = colors[seglabels==foot_label]
    points = points[seglabels == foot_label]
    tri_pcd = trimesh.PointCloud(vertices = points, colors = colors)
    return tri_pcd

def main(base_dir,
         out_dir,
         use_interpenetration=True,
         n_betas=10,
         gender='male',  # male, female, neutral
         viz=True):
    """
    Set up dataset dependent paths to image and joint data, saves results.

    :param base_dir: folder containing LSP images and data
    :param out_dir: output folder
    :param use_interpenetration: boolean, if True enables the interpenetration term
    :param n_betas: number of shape coefficients considered during optimization
    :param use_neutral: boolean, if True enables uses the neutral gender SMPL model
    :param viz: boolean, if True enables visualization during optimization

    dataset:
    """

    input_dir = join(abspath(base_dir), 'input','results-0908','color2depths')
    input_target_dir = join(abspath(base_dir), 'input','results-0908','clean')

    if not exists(out_dir):
        makedirs(out_dir)    
    
    sph_regs = None    
    if gender == 'male':
        model = load_model(MODEL_MALE_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_MALE_PATH)
    elif gender == 'female':
        model = load_model(MODEL_FEMALE_PATH)
        if use_interpenetration:
            sph_regs = np.load(MODEL_FEMALE_PATH)
    else:
        gender == 'neutral'
        model = load_model(MODEL_NEUTRAL_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_NEUTRAL_PATH)

    _LOGGER.info("Reading genders done")
    _LOGGER.info("Loading joints ...")

    # Load joints
    _LOGGER.info("Loading joints done.")

    # Load images


    #Load more
    # 3. SMPLify images
    plyFilePath1 = join(input_target_dir, 'color_to_depth0.ply')
    # meshFilePath1 = join(input_dir, 'mesh.obj')
    jsonFilePath1 = join(input_dir, 'joint0.json')

    plyFilePath2 = join(input_target_dir, 'color_to_depth6.ply')
    # meshFilePath2 = join(input_dir, 'mesh.obj')
    jsonFilePath2 = join(input_dir, 'joint6.json')

    plyFilePath3 = join(input_target_dir, 'color_to_depth1.ply')
    # meshFilePath3 = join(input_dir, 'mesh.obj')
    jsonFilePath3 = join(input_dir, 'joint1.json')

    plyFilePath4 = join(input_target_dir, 'color_to_depth2.ply')
    # meshFilePath4 = join(input_dir, 'mesh.obj')
    jsonFilePath4 = join(input_dir, 'joint2.json')

    plyFilePath5 = join(input_target_dir, 'color_to_depth3.ply')
    # meshFilePath5 = join(input_dir, 'mesh.obj')
    jsonFilePath5 = join(input_dir, 'joint3.json')

    plyFilePath6 = join(input_target_dir, 'color_to_depth4.ply')
    # meshFilePath6 = join(input_dir, 'mesh.obj')
    jsonFilePath6 = join(input_dir, 'joint4.json')

    plyFilePath7 = join(input_target_dir, 'color_to_depth5.ply')
    # meshFilePath7 = join(input_dir, 'mesh.obj')
    jsonFilePath7 = join(input_dir, 'joint5.json')

    if not exists(plyFilePath1):
        print("no file", plyFilePath1)
    if not exists(jsonFilePath1):
        print("no file", jsonFilePath1)
    if not exists(plyFilePath2):
        print("no file", plyFilePath2)
    if not exists(jsonFilePath2):
        print("no file", jsonFilePath2)
    if not exists(plyFilePath3):
        print("no file", plyFilePath3)
    if not exists(jsonFilePath3):
        print("no file", jsonFilePath3)
    if not exists(plyFilePath4):
        print("no file", plyFilePath4)
    if not exists(jsonFilePath4):
        print("no file", jsonFilePath4)

    if not exists(plyFilePath5):
        print("no file", plyFilePath5)
    if not exists(jsonFilePath5):
        print("no file", jsonFilePath5)
    if not exists(plyFilePath6):
        print("no file", plyFilePath6)
    if not exists(jsonFilePath6):
        print("no file", jsonFilePath6)
    if not exists(plyFilePath7):
        print("no file", plyFilePath7)
    if not exists(jsonFilePath7):
        print("no file", jsonFilePath7)

    target1 = trimesh.load(plyFilePath1)
    # target1 = delete_black_and_noise_for_standing_human(target1) # TODO
    target1 = remove_small_clusters(target1)
    target1.vertices /= 1000  # kinect
    # target_mesh1 = trimesh.load(meshFilePath1)
    target2 = trimesh.load(plyFilePath2)
    # target2 = delete_black_and_noise_for_standing_human(target2) # TODO
    target2 = remove_small_clusters(target2)
    target2.vertices /= 1000  # kinect
    # target_mesh2 = trimesh.load(meshFilePath2)
    target3 = trimesh.load(plyFilePath3)
    # target3 = delete_black_and_noise_for_standing_human(target3)  # TODO
    target3 = remove_small_clusters(target3)
    target3.vertices /= 1000  # kinect

    target4 = trimesh.load(plyFilePath4)
    # target4 = delete_black_and_noise_for_standing_human(target4)  # TODO
    target4 = remove_small_clusters(target4)
    target4.vertices /= 1000  # kinect

    target5 = trimesh.load(plyFilePath5)
    # target5 = delete_black_and_noise_for_standing_human(target5)  # TODO
    target5 = remove_small_clusters(target5)
    target5.vertices /= 1000  # kinect

    target6 = trimesh.load(plyFilePath6)
    # target6 = delete_black_and_noise_for_standing_human(target6)  # TODO
    target6 = remove_small_clusters(target6)
    target6.vertices /= 1000  # kinect

    target7 = trimesh.load(plyFilePath7)
    # target7 = delete_black_and_noise_for_standing_human(target7)  # TODO
    target7 = remove_small_clusters(target7)
    target7.vertices /= 1000  # kinect

    with open(jsonFilePath1, "r") as f1:
        jsonstr1 = f1.read()
    joints_json1 = json.loads(jsonstr1)
    j3d1 = joints_json1['people'][0]['pose_keypoints_3d']
    j3d1 = np.array(j3d1).reshape(-1, 3)
    
    j3d1 = j3d1 / 1000 # kinect

    with open(jsonFilePath2, "r") as f2:
        jsonstr2 = f2.read()
    joints_json2 = json.loads(jsonstr2)
    j3d2 = joints_json2['people'][0]['pose_keypoints_3d']
    j3d2 = np.array(j3d2).reshape(-1, 3)

    j3d2 = j3d2 / 1000 # kinect

    with open(jsonFilePath3, "r") as f3:
        jsonstr3 = f3.read()
    joints_json3 = json.loads(jsonstr3)
    j3d3 = joints_json3['people'][0]['pose_keypoints_3d']
    j3d3 = np.array(j3d3).reshape(-1, 3)

    j3d3 = j3d3 / 1000 # kinect

    with open(jsonFilePath4, "r") as f4:
        jsonstr4 = f4.read()
    joints_json4 = json.loads(jsonstr4)
    j3d4 = joints_json4['people'][0]['pose_keypoints_3d']
    j3d4 = np.array(j3d4).reshape(-1, 3)

    j3d4 = j3d4 / 1000 # kinect

    with open(jsonFilePath5, "r") as f5:
        jsonstr5 = f5.read()
    joints_json5 = json.loads(jsonstr5)
    j3d5 = joints_json5['people'][0]['pose_keypoints_3d']
    j3d5 = np.array(j3d5).reshape(-1, 3)

    j3d5 = j3d5 / 1000 # kinect

    with open(jsonFilePath6, "r") as f6:
        jsonstr6 = f6.read()
    joints_json6 = json.loads(jsonstr6)
    j3d6 = joints_json6['people'][0]['pose_keypoints_3d']
    j3d6 = np.array(j3d6).reshape(-1, 3)

    j3d6 = j3d6 / 1000 # kinect

    with open(jsonFilePath7, "r") as f7:
        jsonstr7 = f7.read()
    joints_json7 = json.loads(jsonstr7)
    j3d7 = joints_json7['people'][0]['pose_keypoints_3d']
    j3d7 = np.array(j3d7).reshape(-1, 3)

    j3d7 = j3d7 / 1000 # kinect

    # paramters and projections (viz)
    targets = [target1,target2,target3,target5,target6,target7]
    j3ds = [j3d1,j3d2,j3d3,j3d5,j3d6,j3d7]
    # paramters and projections (viz)
    targets = [target1,target2]
    j3ds = [j3d1,j3d2]

    """"
    #downsampling
    print(type(target1))
    # target1 = trimesh.sample.volume_mesh(target1,100)
    # target2 = trimesh.sample.volume_mesh(target2,100)
    tam1 = o3d.io.read_point_cloud(plyFilePath1)
    tam1 = tam1.voxel_down_sample(voxel_size = 50)
    tam_point1 = np.asarray(tam1.points)
    # target1 = trimesh.PointCloud(tam_point1)
    target1 = trimesh.PointCloud(target1.vertices[np.random.choice(np.arange(len(target1.vertices)), 100)])

    target2 = trimesh.PointCloud(target2.vertices[np.random.choice(np.arange(len(target2.vertices)), 100)])

    target3 = trimesh.PointCloud(target3.vertices[np.random.choice(np.arange(len(target3.vertices)), 100)])

    target1 = trimesh.PointCloud(tam_point1)
    """

    targets = [target1]
    j3ds = [j3d1]

    params_list = run_single_fit(
        targets,
        j3ds,
        model,
        regs=sph_regs,
        n_betas=n_betas,
        viz=viz,
        out_dir=out_dir)

    # show time
    # target, coeff = make_clothes(target_mesh, None, model, params)

    with open(os.path.join(out_dir, 'output.pkl'), 'wb') as outf:  # 'wb' for python 3?
        pickle.dump(params_list, outf)


if __name__ == '__main__':

    """  Parsing the arguments and load the SMPL specific model files    """

    logging.basicConfig(level=logging.INFO)

    base_dir = "."
    out_dir = join(base_dir, 'results')
    use_interpenetration = False
    n_betas = 10
    gender = 'male'
    viz = True
    
    if not use_interpenetration:
        _LOGGER.info('Not using interpenetration term.')

    # 1. load SMPL models (independent upon dataset)
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(base_dir, 'models')
    # Model paths:
    MODEL_NEUTRAL_PATH = join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    if use_interpenetration:
        # paths to the npz files storing the regressors for capsules
        SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR,
                                     'regressors_locked_normalized_hybrid.npz')
        SPH_REGS_FEMALE_PATH = join(MODEL_DIR,
                                    'regressors_locked_normalized_female.npz')
        SPH_REGS_MALE_PATH = join(MODEL_DIR,
                                  'regressors_locked_normalized_male.npz')

    # 3. call the  main function
    main(base_dir, out_dir, use_interpenetration, n_betas, gender, viz)
