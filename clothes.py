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

# basics
import numpy as np
import chumpy as ch
from scipy.spatial import cKDTree

from utils import *

# SMPL
#from smpl_webuser.verts import verts_decorated
from smpl.verts import verts_decorated

import smpl_pyrender
import pyrender

from playbvh.smplclothbvhplayer import playBVH

def make_clothes(target,
                label,
                model,
                init=None,
                viz=True):    

    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(10)

    # initialize the pose by using the optimized body orientation and the
    # pose prior
    init_pose = ch.zeros(72)

    # instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)
    sv = verts_decorated(
        scale=ch.ones(1), # GU; result = scale * result + tr; result.scale = scale (verts_decorated, verts.py)
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :10],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)
    
    if init:
        sv.scale[:] = init["scale"][:]
        sv.trans[:] = init["trans"][:]
        sv.betas[:] = init["betas"][:]
        sv.pose[:] = init["pose"][:]
    
    sv_tree = cKDTree(sv[:])
    axis_x, axis_y, axis_z = setup_vertex_local_coord(sv.f, sv.r[:]) # (6890, 3)

    cps_t2m = sv_tree.query(target.vertices, k=1, p=2, n_jobs=-1)[1]
    
    displacement = target.vertices[:] - sv[cps_t2m]
    coeff_x = np.einsum('ij,ij->i', axis_x[cps_t2m], displacement)
    coeff_y = np.einsum('ij,ij->i', axis_y[cps_t2m], displacement)
    coeff_z = np.einsum('ij,ij->i', axis_z[cps_t2m], displacement)
    coeff_z[coeff_z < 0.01] = 0.01 # 메쉬가 모델 내부로 파고드는 것 방지

    coeff = np.stack([coeff_x, coeff_y, coeff_z], axis=-1)

    if viz:
        cam_pose = np.array([
            [1.0,  0.0,  0.0,  0.0],   # zero-rotation  
            [0.0, -1.0,  0.0, -0.5],   # 
            [0.0,  0.0, -1.0,  0.0],   # (0, 0, 2.0) translation  
            [0.0,  0.0,  0.0,  1.0]
        ])
        scene, viewer = smpl_pyrender.setupScene(cam_pose)
        smpl_pyrender.smplMeshNode = None
        global targetNode
        targetNode = None

        def update_target_pcd():
            # FIXME GPU
            axis = np.stack(setup_vertex_local_coord(sv.f, sv.r[:]), axis=-1)            
            target.vertices = sv[cps_t2m] + np.einsum('ij,ikj->ik', coeff, axis[cps_t2m])

        def on_step():

            global targetNode

            smpl_pyrender.updateSMPL(scene, viewer, sv[:], sv.f)
            viewer.render_lock.acquire()        
            if targetNode is not None:
                scene.remove_node(targetNode)
            targetNode = scene.add(pyrender.Mesh.from_trimesh(target, smooth=True))
            viewer.render_lock.release()

        update_target_pcd()
        on_step()

        def callback():
            update_target_pcd()
            on_step()

        input("start?")
        playBVH(sv.pose, "./playbvh/walking.bvh", callback)
        input("finish?")
        viewer.close_external()

    return target, coeff