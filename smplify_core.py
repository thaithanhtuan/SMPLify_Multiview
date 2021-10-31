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
Modification : 9

For Windows Enviroment, we use only Camera module from OpenDR and replay the Rendering with PyRedner


dependent 
===============

numpy    (pip install numpy) 
opencv   (pip install opencv-python)
chumpy   (pip install chumpy)
smpl     (modified by seoultech)
opendr   (modified by SeoulTech) 
smplify  (named for package) 


"""
import os
import logging
# import cPickle as pickle   # Python 2
import _pickle as pickle   # Python 3
from time import time

# basics
import numpy as np
import chumpy as ch
from scipy.spatial import cKDTree

from utils import *


# SMPL
from smpl.lbs import global_rigid_transformation
#from smpl_webuser.verts import verts_decorated
from smpl.verts import verts_decorated

# SMPLIFY
from smplify.robustifiers import GMOf
#from lib.sphere_collisions import SphereCollisions
from smplify.sphere_collisions import SphereCollisions
#from lib.max_mixture_prior import MaxMixtureCompletePrior
from smplify.max_mixture_prior import MaxMixtureCompletePrior

# Trimesh
import trimesh

_LOGGER = logging.getLogger(__name__)

# Mapping from Kinnect joints to SMPL joints.
# 24    Right ankle         8
# 23    Right knee          5
# 22    Right hip           2
# 18    Left hip            1
# 19    Left knee           4
# 20    Left ankle          7
# 14    Right wrist         21
# 13    Right elbow         19
# 12    Right shoulder      17
# 5     Left shoulder       16
# 6     Left elbow          18
# 7     Left wrist          20

# 15    Right hand          21
# 8     Left hand           22
# 25    Right foot          11
# 21    Left foot           10
# 3     Neck                12
# 26    Head                15

# --------------------Core optimization --------------------
def optimize_j3d_v2(j3ds,
                model,
                prior,
                n_betas=10,
                targets=None,
                regs=None,
                conf=None,
                viz=False,
                cam_pose=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param j3ds: list of 14x3 array of CNN joints
    :param model: SMPL model
    :param prior: mixture of gaussians pose prior
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param conf: 18D vector storing the confidence values from the CNN
    :param viz: boolean, if True enables visualization during optimization
    :returns: the optimized model
    """
    use_scale = True
    t0 = time()
    # define the mapping Kinnect joints -> SMPL joints
    # cids are joints ids for Kinnect:
    cids = [24, 23, 22, 18, 19, 20, 14, 13, 12,
            5, 6, 7, 15, 8, 25, 21, 3, 26]

    # joint ids for SMPL
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17,
                16, 18, 20, 21, 22, 11, 10, 12, 15]
    #Thai edit left right
    #smpl_ids = [7, 4, 1, 2, 5, 8, 20, 18, 16,
    #            17, 19, 21, 22, 21, 10, 11, 12, 15]
    # Mapping from Kinnect joints to SMPL joints.
    # 24    Right ankle         7 '
    # 23    Right knee          4 '
    # 22    Right hip           1 '
    # 18    Left hip            2 '
    # 19    Left knee           5 '
    # 20    Left ankle          8 '
    # 14    Right wrist         20 '
    # 13    Right elbow         18 '
    # 12    Right shoulder      16 '
    # 5     Left shoulder       17 '
    # 6     Left elbow          19 '
    # 7     Left wrist          21 '

    # 15    Right hand          22 '
    # 8     Left hand           21 '
    # 25    Right foot          10 '
    # 21    Left foot           11 '
    # 3     Neck                12
    # 26    Head                15
    # weights assigned to each joint during optimization;
    # the definition of hips in SMPL and Kinnect is significantly different so set
    base_weights = np.array(
        [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1,  2, 1, 1, 1, 2, 2], dtype=np.float64)
    # [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    w_j3d = 10
    w_pose_prior = 1
    w_sph_coll = 1e3
    w_pose_exp = 1
    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(n_betas)

    # initialize the pose by using the optimized body orientation and the
    # pose prior
    init_poses = []
    for i in range(len(j3ds)):
        init_poses.append(np.hstack(([0,0,1], prior.weights.dot(prior.means))))

    # instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)
    svs = []
    for i in range(len(j3ds)):

        svs.append(verts_decorated(
            scale=ch.ones(1),
            trans=ch.zeros(3),
            pose=ch.array(init_poses[i]),
            v_template=model.v_template,
            J=model.J_regressor,
            betas=betas,
            shapedirs=model.shapedirs[:, :, :n_betas],
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style=model.bs_style,
            f=model.f,
            bs_type=model.bs_type,
            posedirs=model.posedirs))

    # make the SMPL joints depend on betas
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                        for i in range(len(betas))])
    J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(model.v_template.r)

    # get joint positions as a function of model pose, betas, scale and trans
    Jtrs = []
    for i in range(len(j3ds)):

        (_, A_global) = global_rigid_transformation(svs[i].pose, J_onbetas, model.kintree_table, xp=ch)
        Jtr = svs[i].scale * ch.vstack([g[:3, 3] for g in A_global]) + svs[i].trans
        Jtrs.append(Jtr)

    # update the weights using confidence values
    weights = base_weights * conf[cids] if conf is not None else base_weights

    # obj1. data term: distance between observed and estimated joints in 2D
    ###########################

    def obj_j3d(j3d, w, sigma, Jtr):
        return (w * weights.reshape((-1, 1)) *  GMOf((j3d[cids] - Jtr[smpl_ids]), sigma))

    # obj2: mixture of gaussians pose prior
    ###########################

    def pprior(w, sv): return w * prior(sv.pose)
    # obj3: joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10
    def my_exp(x): return alpha * ch.exp(x)

    def obj_angle(w, sv): return w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
        58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])


    viz = True
    j3d_loss = []
    pose_prior_loss = []
    pose_exp_loss = []
    pose_beta_loss = []
    if viz is True:
        import smpl_pyrender
        import pyrender
        id = 0
        print("-----------------------------id:", id)
        if cam_pose is None:
            cam_pose = np.array([
                [ 1.0,  0.0,  0.0,  0.0],   # zero-rotation
                [ 0.0, -1.0,  0.0,  0.0],   #
                [ 0.0,  0.0, -1.0, -1.0],   # (0, 0, 2.0) translation
                [ 0.0,  0.0,  0.0,  1.0]
            ])
        scene, viewer = smpl_pyrender.setupScene(cam_pose)

        if type(targets[id]) == trimesh.base.Trimesh:
            target_pyrender = pyrender.Mesh.from_trimesh(targets[id], smooth = False)
        if type(targets[id]) == trimesh.points.PointCloud:
            target_pyrender = pyrender.Mesh.from_points(targets[id].vertices, targets[id].colors)

        smpl_pyrender.smplMeshNode = None

        # show joints
        sm = trimesh.creation.uv_sphere(radius=0.02)
        sm.visual.vertex_colors = [1.0, 1.0, 0.0]
        tfs = np.tile(np.eye(4), (len(j3ds[id][cids]), 1, 1))
        tfs[:,:3,3] = j3ds[id][cids]
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        # show joints

        global sm11, iter_num
        sm11 = None
        iter_num = 0
        def on_step(_):
            global sm11, iter_num
            iter_num += 1
            print("iternum:", iter_num)
            # print error
            print("Step error:")
            if ('objs' in locals()):
                print("error-j3d:", np.sum(objs['j3d_'+str(i)].r**2))
                j3d_loss.append(np.sum(objs['j3d_'+str(i)].r**2))
                print("error-pose prior:", np.sum(objs['pose_'+str(i)].r ** 2))
                pose_prior_loss.append(np.sum(objs['pose_'+str(i)].r ** 2))

                if stage != 4:
                    print("error-pose exp:", np.sum(objs['pose_exp_' + str(i)].r ** 2))
                    pose_exp_loss.append(np.sum(objs['pose_exp_'+str(i)].r ** 2))
                else:
                    pose_exp_loss.append(0.0)
                print("error-betas:", np.sum(objs['betas'].r ** 2))
                pose_beta_loss.append(np.sum(objs['betas'].r ** 2))
            try:
                smpl_pyrender.updateSMPL(scene, viewer, svs[id][:], svs[id].f)
            except:
                pass
            print("iternum:", iter_num, "updatesmpl")
            if sm11 is not None:
                scene.remove_node(sm11)
                print("iternum:", iter_num, "remove")
            print("iternum:", iter_num, "1st")
            if(iter_num > 27):
                sm1 = trimesh.creation.uv_sphere(radius=0.02)
                sm1.visual.vertex_colors = [0.0, 0.0, 1.0]
                tfs1 = np.tile(np.eye(4), (len(Jtrs[id][smpl_ids]), 1, 1))
                tfs1[:, :3, 3] = Jtrs[id][smpl_ids]

                m1 = pyrender.Mesh.from_trimesh(sm1, poses=tfs1)

                point = o3d.geometry.PointCloud()
                point.points = o3d.utility.Vector3dVector(Jtrs[id][smpl_ids])
                point.paint_uniform_color((1, 0, 0))

                point2 = o3d.geometry.PointCloud()
                point2.points = o3d.utility.Vector3dVector(j3ds[id][cids])
                point2.paint_uniform_color((0, 1, 0))

                pointSMPL = o3d.geometry.PointCloud()
                pointSMPL.points = o3d.utility.Vector3dVector(svs[id][:])
                pointSMPL.paint_uniform_color((0, 0, 1))

                point3 = o3d.geometry.PointCloud()
                point3 = point + point2 + pointSMPL
                point3.points = o3d.utility.Vector3dVector(np.asarray(point3.points)* 1000)
                o3d.visualization.draw_geometries([point3], width=600, height=600)

                o3d.io.write_point_cloud("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".ply", point3, write_ascii = True)
                # sm11.save_gif("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".gif")

                print("iternum:", iter_num, "2nd")
                viewer.render_lock.acquire()
                scene.add(m)
                sm11 = scene.add(m1)
                viewer.render_lock.release()
                print("iternum:", iter_num, "3rd")
                if targets[id] is not None:
                    viewer.render_lock.acquire()
                    scene.add(target_pyrender)
                    # sm11.save_gif("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".gif")
                    # sm11.save_gif("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".gif")
                    viewer.render_lock.release()



            print("Iternum:", iter_num, "finished")
            means = []
            distances = []
            for j in range(len(j3ds)):
                distances.append(np.sqrt(np.sum(np.square((j3ds[j][cids] - Jtrs[j][smpl_ids])),axis=1))*1000)
                means.append(np.mean(np.sqrt(np.sum(np.square((j3ds[j][cids] - Jtrs[j][smpl_ids])),axis=1))*1000))
            print("Distances:",distances)
            print("Means:",means)
            print("Average",np.mean(means))

    else:
        on_step = None

    # obj5: interpenentration
    ###########################
    if regs is not None:
        # interpenetration term
        sps = []
        for j in range(len(j3ds)):
            sp = SphereCollisions(
                pose=svs[j].pose, betas=svs[j].betas, model=model, regs=regs)
            sp.no_hands = True
            sps.append(sp)

    #############################################
    # 5. optimize
    #############################################
    # weight configuration used in the paper, with joints + confidence values from the CNN
    # (all the weights used in the code were obtained via grid search, see the paper for more details)
    # the first list contains the weights for the pose priors,
    # the second list contains the weights for the shape prior
    opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78, 0.75, 0 ],
                        [1e2, 5 * 1e1, 1e1, .5 * 1e1, .5 * 1e1])
    # opt_weights = zip([0.75, 0.75, 0.75, 0.75, 0.75],
    #                     [.5 * 1e1, .5 * 1e1, .5 * 1e1, .5 * 1e1, .5 * 1e1])
    # run the optimization in 4 stages, progressively decreasing the
    # weights for the priors
    for stage, (w, wbetas) in enumerate(opt_weights):
        _LOGGER.info('stage %01d', stage)
        objs = {}
        for i in range(len(j3ds)):

            objs['j3d_'+str(i)] = w_j3d * obj_j3d(j3ds[i], 1., 100, Jtrs[i])
            # objs['pose_'+str(i)] = w_pose_prior * pprior(w, svs[i])
            if stage != 4:
                objs['pose_exp_'+str(i)] = w_pose_exp * obj_angle(0.317 * w, svs[i])

            if regs is not None:#1123456789
                objs['sph_coll_'+str(i)] = w_sph_coll * sps[i]
        objs['betas'] = len(j3ds) * wbetas * betas

        t = []
        p = []
        freevariables = [betas]
        for i in range(len(j3ds)):
            print(type(betas))
            print(type(svs[i].scale))
            if(stage == 0):
                freevariables += [svs[i].trans, svs[i].pose]
            else:
                if (use_scale):
                    freevariables += [svs[i].scale, svs[i].trans, svs[i].pose]
                else:
                    freevariables += [svs[i].trans, svs[i].pose]

            # freevariables += [svs[i].scale, svs[i].trans, svs[i].pose]
        Optimized = ch.minimize(
            objs,                   # objective functions
            x0=freevariables,  # free-variables
            method='dogleg',
            callback=on_step,
            options={'maxiter': 100, 'e_3': .0001, 'disp': 0})


        """else:
            s, t, b, p = ch.minimize(
                objs,                   # objective functions
                x0=[sv.scale, sv.trans, sv.betas, sv.pose],  # free-variables
                method='dogleg',
                callback=on_step,
                options={'maxiter': 100, 'e_3': .0001, 'disp': 0})"""

    print("ALL loss over iteration:")
    print("error-j3d:", j3d_loss)
    print("error-pose prior:", pose_prior_loss)
    print("error-pose exp:", pose_exp_loss)
    print("error-beta:", pose_beta_loss)
    """
    point = o3d.geometry.PointCloud()
    point.points = o3d.utility.Vector3dVector(Jtrs[id][smpl_ids])
    point.paint_uniform_color((1, 0, 0))

    point2 = o3d.geometry.PointCloud()
    point2.points = o3d.utility.Vector3dVector(j3ds[id][cids])
    point2.paint_uniform_color((1, 1, 0))

    o3d.visualization.draw_geometries([point,point2])
    """


    t1 = time()
    _LOGGER.info('elapsed %.05f', (t1 - t0))

    """if viz is True:
        viewer.close_external()"""
    
    return svs


# --------------------Core optimization --------------------
def optimize_icp_v2(targets,
                 model,
                 prior,
                 n_betas=10,
                 regs=None,
                 init=None,
                 maxiter=30,
                 viz=False,
                 save_dir=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param target: trimesh mesh or pointcloud
    :param model: SMPL model
    :param prior: mixture of gaussians pose prior
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param init: initial parameter dictionary
    :param maxiter: maxiter
    :param viz: boolean, if True enables visualization during optimization
    :returns: the optimized model
    """
    use_scale = False
    t0 = time()

    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(n_betas)

    # initialize the pose by using the optimized body orientation and the
    # pose prior
    init_poses = []
    for i in range(len(targets)):
        init_poses.append(np.hstack(([0, 0, 1], prior.weights.dot(prior.means))))

    # instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)

    svs = []
    for i in range(len(targets)):
        svs.append(verts_decorated(
            scale=ch.ones(1),
            trans=ch.zeros(3),
            pose=ch.array(init_poses[i]),
            v_template=model.v_template,
            J=model.J_regressor,
            betas=betas,
            shapedirs=model.shapedirs[:, :, :n_betas],
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style=model.bs_style,
            f=model.f,
            bs_type=model.bs_type,
            posedirs=model.posedirs))

    target_trees = []
    target_normals = []
    for i in range(len(targets)):
        target_tree = cKDTree(targets[i].vertices)
        target_normal = estimate_normals(targets[i].vertices, radius=40, orient=[0, 0, -10])

        target_trees.append(target_tree)
        target_normals.append(target_normal)

    def pprior(w, sv):
        return w * prior(sv.pose)

    # obj3: joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10

    def my_exp(x):
        return alpha * ch.exp(x)

    def obj_angle(w, sv):
        return w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
            58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])

    if regs is not None:
        # interpenetration term
        sps = []
        for i in range(len(targets)):
            sp = SphereCollisions(
                pose=svs[i].pose, betas=svs[i].betas, model=model, regs=regs)
            sp.no_hands = True
            sps.append(sp)

    #############################################
    # 5. optimize
    #############################################
    viz = True
    if viz is True:
        import smpl_pyrender
        import pyrender
        id = 0
        print("-----------------------------id:", id)
        # global cam_pose = None

        cam_pose = np.array([
            [ 1.0,  0.0,  0.0,  0.0],   # zero-rotation
            [ 0.0, -1.0,  0.0,  0.0],   #
            [ 0.0,  0.0, -1.0, -1.0],   # (0, 0, 2.0) translation
            [ 0.0,  0.0,  0.0,  1.0]
        ])
        scene, viewer = smpl_pyrender.setupScene(cam_pose)

        if type(targets[id]) == trimesh.base.Trimesh:
            target_pyrender = pyrender.Mesh.from_trimesh(targets[id], smooth = False)
        if type(targets[id]) == trimesh.points.PointCloud:
            target_pyrender = pyrender.Mesh.from_points(targets[id].vertices, targets[id].colors)

        smpl_pyrender.smplMeshNode = None

        """
        # show joints
        sm = trimesh.creation.uv_sphere(radius=0.02)
        sm.visual.vertex_colors = [1.0, 1.0, 0.0]
        tfs = np.tile(np.eye(4), (len(j3ds[id][cids]), 1, 1))
        tfs[:,:3,3] = j3ds[id][cids]
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        """


        # show joints

        global sm11, iter_num
        sm11 = None
        iter_num = 0
        def on_step(_):
            global sm11, iter_num
            iter_num += 1
            print("iternum:", iter_num)

            #print error
            print("Step error:")
            if('objs' in locals()):
                print("error-j3d:", objs['dist_m2t_'+str(j)])

            try:
                smpl_pyrender.updateSMPL(scene, viewer, svs[id][:], svs[id].f)
            except:
                pass
            print("iternum:", iter_num, "updatesmpl")
            if sm11 is not None:
                scene.remove_node(sm11)
                print("iternum:", iter_num, "remove")

            """
            print("iternum:", iter_num, "1st")
            
            sm1 = trimesh.creation.uv_sphere(radius=0.02)
            sm1.visual.vertex_colors = [0.0, 0.0, 1.0]
            tfs1 = np.tile(np.eye(4), (len(Jtrs[id][smpl_ids]), 1, 1))
            tfs1[:, :3, 3] = Jtrs[id][smpl_ids]
            m1 = pyrender.Mesh.from_trimesh(sm1, poses=tfs1)

            point = o3d.geometry.PointCloud()
            point.points = o3d.utility.Vector3dVector(Jtrs[id][smpl_ids])
            point.paint_uniform_color((1, 0, 0))

            point2 = o3d.geometry.PointCloud()
            point2.points = o3d.utility.Vector3dVector(j3ds[id][cids])
            point2.paint_uniform_color((0, 1, 0))

            pointSMPL = o3d.geometry.PointCloud()
            pointSMPL.points = o3d.utility.Vector3dVector(svs[id][:])
            pointSMPL.paint_uniform_color((0, 0, 1))

            point3 = o3d.geometry.PointCloud()
            point3 = point + point2 + pointSMPL
            point3.points = o3d.utility.Vector3dVector(np.asarray(point3.points)* 1000)
            o3d.visualization.draw_geometries([point3], width=600, height=600)

            o3d.io.write_point_cloud("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".ply", point3, write_ascii = True)
            # sm11.save_gif("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".gif")

            print("iternum:", iter_num, "2nd")
            viewer.render_lock.acquire()
            scene.add(m)
            sm11 = scene.add(m1)
            viewer.render_lock.release()
            print("iternum:", iter_num, "3rd")
            """

            if targets[id] is not None:
                viewer.render_lock.acquire()
                scene.add(target_pyrender)
                # sm11.save_gif("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".gif")
                # sm11.save_gif("./results/Render/ID_" + str(i) + "_Iter_" + str(iter_num) + ".gif")
                viewer.render_lock.release()

            """
            #Calculate joint distance
            print("Iternum:", iter_num, "finished")
            means = []
            distances = []
            for j in range(len(j3ds)):
                distances.append(np.sqrt(np.sum(np.square((j3ds[j][cids] - Jtrs[j][smpl_ids])),axis=1))*1000)
                means.append(np.mean(np.sqrt(np.sum(np.square((j3ds[j][cids] - Jtrs[j][smpl_ids])),axis=1))*1000))
            print("Distances:",distances)
            print("Means:",means)
            print("Average",np.mean(means))
            """


    else:
        on_step = None


    """
        if viz is True:

        import smpl_pyrender
        import pyrender

        cam_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],  # zero-rotation
            [0.0, -1.0, 0.0, -0.5],  #
            [0.0, 0.0, -1.0, 0.0],  # (0, 0, 2.0) translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene, viewer = smpl_pyrender.setupScene(cam_pose)

        if type(target) == trimesh.base.Trimesh:
            target_pyrender = pyrender.Mesh.from_trimesh(target, smooth=False)
        if type(target) == trimesh.points.PointCloud:
            target_pyrender = pyrender.Mesh.from_points(target.vertices, target.colors)

        smpl_pyrender.smplMeshNode = None

        def on_step(_):
            smpl_pyrender.updateSMPL(scene, viewer, sv[:], sv.f)

            viewer.render_lock.acquire()
            scene.add(target_pyrender)
            viewer.render_lock.release()

    else:
        on_step = None
    """




    if init is not None:
        svs = init["svs"]

    for i in range(maxiter):
        objs = {}
        _LOGGER.info('Iteration %01d', i)
        for j in range(len(targets)):
            sv_tree = cKDTree(svs[j][:])
            _, sv_normal = calc_normal_vectors(svs[j][:], svs[j].f)

            cps_m2t = target_trees[j].query(svs[j][:], k=1, p=2)[1]  # (dist, idx)
            cps_t2m = sv_tree.query(targets[j].vertices, k=1, p=2, n_jobs=-1)[1]

            def dist_m2t(thres=0):
                mask = np.sum(sv_normal[:] * target_normals[j][cps_m2t], axis=-1) > thres  # 노말벡터가 비슷한 방향인 경우
                distv = svs[j][:] - targets[j].vertices[cps_m2t]
                dist = ch.sum(distv ** 2, axis=-1) ** 0.5
                return dist[mask] ** 2 + ch.abs(dist[mask])

            def dist_t2m(thres=0):
                mask = np.sum(sv_normal[cps_t2m] * target_normals[j][:], axis=-1) > thres  # 노말벡터가 비슷한 방향인 경우
                distv = svs[j][cps_t2m] - targets[j].vertices[:]
                dist = ch.sum(distv ** 2, axis=-1) ** 0.5
                return dist[mask] ** 2 + ch.abs(dist[mask])

            def penalty_m2t(w, thres=0):
                distv = svs[j][:] - targets[j].vertices[cps_m2t]
                mask = np.sum(target_normals[j][cps_m2t] * distv.r, axis=-1) > thres
                dist = w * ch.sum(distv ** 2, axis=-1) ** 0.5
                return ch.exp(dist[mask]) / w  # 노말벡터 기준으로 모델이 타겟보다 앞에 있을 경우 엄청난 페널티

            def penalty_t2m(w, thres=0):
                distv = svs[j][cps_t2m] - targets[j].vertices[:]
                mask = np.sum(target_normals[j][:] * distv.r, axis=-1) > thres
                dist = w * ch.sum(distv ** 2, axis=-1) ** 0.5
                return ch.exp(dist[mask]) / w  # 노말벡터 기준으로 모델이 타겟보다 앞에 있을 경우 엄청난 페널티

            w = 0.75 * 1e-2  # 0.75
            wbetas = 0.5 * 1e-1  # 5.0


            objs['dist_m2t_'+str(j)] = dist_m2t() / 2
            objs['dist_t2m_'+str(j)] = dist_t2m() / 2
            objs['penalty_m2t_'+str(j)] = penalty_m2t(10) / 2
            objs['penalty_t2m_'+str(j)] = penalty_t2m(10) / 2
            objs['pose_'+str(j)] = pprior(w, svs[j])
            # objs['pose_exp'] = obj_angle(0.317 * w)

            if regs is not None:
                objs['sph_coll_'+str(j)] = 1e-1 * sps[j]

        objs['betas'] = wbetas * betas

        freevariables = [betas]
        for j in range(len(targets)):
            if (use_scale):
                freevariables += [svs[j].scale, svs[j].trans, svs[j].pose]
            else:
                freevariables += [svs[j].trans, svs[j].pose]


        Optimized = ch.minimize(
            objs,  # objective functions
            x0=freevariables,  # free-variables
            method='dogleg',
            callback=on_step,
            options={'maxiter': 100, 'e_3': .0001, 'disp': 0})



        if save_dir is not None:
            with open(os.path.join(save_dir, 'output' + str(i) + '.pkl'), 'wb') as outf:  # 'wb' for python 3?
                pickle.dump(svs, outf)
            with open(os.path.join(save_dir, 'output_icp.pkl'), 'wb') as outf:  # 'wb' for python 3?
                pickle.dump(svs, outf)

    """
        if viz is True:
        viewer.close_external()

    point = o3d.geometry.PointCloud()
    point.points = o3d.utility.Vector3dVector(Jtrs[id][smpl_ids])
    point.paint_uniform_color((1, 0, 0))

    point2 = o3d.geometry.PointCloud()
    point2.points = o3d.utility.Vector3dVector(j3ds[id][cids])
    point2.paint_uniform_color((1, 1, 0))

    o3d.visualization.draw_geometries([point, point2])
    """


    t1 = time()
    _LOGGER.info('elapsed %.05f', (t1 - t0))

    """if viz is True:
        viewer.close_external()"""

    return svs

# --------------------Core optimization --------------------
def optimize_icp(target,
                model,
                prior,
                n_betas=10,
                regs=None,
                init=None,
                maxiter=30,
                viz=False,
                save_dir=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param target: trimesh mesh or pointcloud
    :param model: SMPL model
    :param prior: mixture of gaussians pose prior
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param init: initial parameter dictionary
    :param maxiter: maxiter
    :param viz: boolean, if True enables visualization during optimization
    :returns: the optimized model
    """

    t0 = time()

    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.zeros(n_betas)

    # initialize the pose by using the optimized body orientation and the
    # pose prior
    init_pose = np.hstack(([0,0,0], prior.weights.dot(prior.means)))

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
        shapedirs=model.shapedirs[:, :, :n_betas],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    target_tree = cKDTree(target.vertices)
    target_normal = estimate_normals(target.vertices, radius=40, orient=[0,0,-10])

    def pprior(w): return w * prior(sv.pose)
    # obj3: joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10
    def my_exp(x): return alpha * ch.exp(x)

    if regs is not None:
        # interpenetration term
        sp = SphereCollisions(
            pose=sv.pose, betas=sv.betas, model=model, regs=regs)
        sp.no_hands = True    

    #############################################
    # 5. optimize
    #############################################
    
    if viz is True:
        
        import smpl_pyrender
        import pyrender

        cam_pose = np.array([
            [1.0,  0.0,  0.0,  0.0],   # zero-rotation  
            [0.0,  -1.0, 0.0, -0.5],   # 
            [0.0,  0.0, -1.0,  0.0],   # (0, 0, 2.0) translation  
            [0.0,  0.0,  0.0,  1.0]
        ])
        scene, viewer = smpl_pyrender.setupScene(cam_pose)

        if type(target) == trimesh.base.Trimesh:
            target_pyrender = pyrender.Mesh.from_trimesh(target, smooth=False)
        if type(target) == trimesh.points.PointCloud:
            target_pyrender = pyrender.Mesh.from_points(target.vertices, target.colors)
        
        smpl_pyrender.smplMeshNode = None
        
        def on_step(_):
            smpl_pyrender.updateSMPL(scene, viewer, sv[:], sv.f)

            viewer.render_lock.acquire()
            scene.add(target_pyrender)
            viewer.render_lock.release()

    else:
        on_step = None

    if init is not None:
        sv.scale[:] = init["scale"][:]
        sv.trans[:] = init["trans"][:]
        sv.betas[:] = init["betas"][:]
        sv.pose[:] = init["pose"][:]
    
    for i in range(maxiter):

        _LOGGER.info('Iteration %01d', i)

        sv_tree = cKDTree(sv[:])
        _, sv_normal = calc_normal_vectors(sv[:], sv.f)

        cps_m2t = target_tree.query(sv[:], k=1, p=2)[1]  # (dist, idx)
        cps_t2m = sv_tree.query(target.vertices, k=1, p=2, n_jobs=-1)[1]
        
        def dist_m2t(thres=0):
            mask = np.sum(sv_normal[:] * target_normal[cps_m2t], axis=-1) > thres # 노말벡터가 비슷한 방향인 경우
            distv = sv[:] - target.vertices[cps_m2t]
            dist = ch.sum(distv**2, axis=-1)**0.5
            return dist[mask]**2 + ch.abs(dist[mask])

        def dist_t2m(thres=0):
            mask = np.sum(sv_normal[cps_t2m] * target_normal[:], axis=-1) > thres # 노말벡터가 비슷한 방향인 경우
            distv = sv[cps_t2m] - target.vertices[:]
            dist = ch.sum(distv**2, axis=-1)**0.5
            return dist[mask]**2 + ch.abs(dist[mask])

        def penalty_m2t(w, thres=0):
            distv = sv[:] - target.vertices[cps_m2t]
            mask = np.sum(target_normal[cps_m2t] * distv.r, axis=-1) > thres
            dist = w * ch.sum(distv**2, axis=-1)**0.5
            return ch.exp(dist[mask]) / w # 노말벡터 기준으로 모델이 타겟보다 앞에 있을 경우 엄청난 페널티

        def penalty_t2m(w, thres=0):
            distv = sv[cps_t2m] - target.vertices[:]
            mask = np.sum(target_normal[:] * distv.r, axis=-1) > thres
            dist = w * ch.sum(distv**2, axis=-1)**0.5
            return ch.exp(dist[mask]) / w # 노말벡터 기준으로 모델이 타겟보다 앞에 있을 경우 엄청난 페널티

        w = 0.75 * 1e-1 # 0.75
        wbetas = 0.5 * 1e0 # 5.0

        objs = {}
        objs['dist_m2t'] = dist_m2t() / 2
        objs['dist_t2m'] = dist_t2m() / 2
        objs['penalty_m2t'] = penalty_m2t(10) / 2
        objs['penalty_t2m'] = penalty_t2m(10) / 2
        objs['pose'] = pprior(w)
        # objs['pose_exp'] = obj_angle(0.317 * w)
        objs['betas'] = wbetas * betas
        if regs is not None:
            objs['sph_coll'] = 1e-1 * sp

        s, t, b, p = ch.minimize(
            objs,                   # objective functions
            x0=[sv.trans, sv.betas, sv.pose],  # free-variables
            method='dogleg',
            callback=on_step,
            options={'maxiter': 1, 'e_3': .0001, 'disp': 0})
        
        # checking optimized pose and shape
        print('   scale :', s)
        print('   trans :', t)
        print('   betas;', b)
        print('   pose :', p.reshape((-1, 3))[0])
        
        params = {'scale': sv.scale.r,
                  'trans': sv.trans.r,
                  'betas': sv.betas.r,
                  'pose': sv.pose.r}

        if save_dir is not None:
            with open(os.path.join(save_dir, 'output' + str(i) + '.pkl'), 'wb') as outf:  # 'wb' for python 3?
                pickle.dump(params, outf)
            with open(os.path.join(save_dir, 'output_icp.pkl'), 'wb') as outf:  # 'wb' for python 3?
                pickle.dump(params, outf)
    
    if viz is True:
        viewer.close_external()

    t1 = time()
    _LOGGER.info('elapsed %.05f', (t1 - t0))

    return sv

"""def run_single_fit(target,
                   j3d,
                   model,
                   regs=None,
                   n_betas=10,
                   viz=False,  # optimize_on_joints
                   out_dir=None):
    Run the fit for one specific image.
    :param target: trimesh pcd
    :param j3d: 3-D joints coordinate, N * 3 array
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing camera/model parameters and images with rendered fits
    

    ###################################
    # 1. prior setting
    ###################################
    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))
    print(' gmm: w=', prior.weights.r)   # pose GMM statistics chumpy
    print(' gmm: m=', prior.means.shape)  # pose GMM statistics numpy
    print(" inital pose:", init_pose.reshape((-1, 3)))  # numpy

    ###################################
    # 3. fit
    ####################################
    
    choice = 'n'
    if os.path.isfile(os.path.join(out_dir, 'output_j3d.pkl')):
        choice = input("skip j3d? [y/n] ")

    if choice == 'n' or choice == 'N':
        sv = optimize_j3d(
            j3d,
            model,   
            prior,  # priors
            n_betas=n_betas,  # shape params size
            target=target,
            viz=viz,     # visualizing or not
            regs=regs
        )

        params = {'scale': sv.scale.r,
                    'trans': sv.trans.r,
                    'betas': sv.betas.r,
                    'pose': sv.pose.r }

        with open(os.path.join(out_dir, 'output_j3d.pkl'), 'wb') as outf:  # 'wb' for python 3?
            pickle.dump(params, outf)
    else:
        with open(os.path.join(out_dir, 'output_j3d.pkl'), 'rb') as p:
            params = pickle.load(p)    
    
    choice = 'n'
    if os.path.isfile(os.path.join(out_dir, 'output_icp.pkl')):
        choice = input("skip icp? [y/n] ")

    if choice == 'n' or choice == 'N':
        sv = optimize_icp(
            target,
            model,   
            prior,  # priors
            n_betas=n_betas,  # shape params size
            maxiter=30,            
            regs=regs,
            init=params,
            viz=viz,     # visualizing or not
            save_dir=out_dir
        )
        
        # 5. return resultant fit parameters  (pose, shape) is what we want but camera needed
        # save all Camera parameters and SMPL paramters
        params = {'scale': sv.scale.r,
                    'trans': sv.trans.r,
                    'betas': sv.betas.r,
                    'pose': sv.pose.r }
        
        with open(os.path.join(out_dir, 'output_icp.pkl'), 'wb') as outf:  # 'wb' for python 3?
            pickle.dump(params, outf)
    
    else:
        with open(os.path.join(out_dir, 'output_icp.pkl'), 'rb') as p:
            params = pickle.load(p)

    return params
"""

def run_single_fit(targets,
                   j3ds,
                   model,
                   regs=None,
                   n_betas=10,
                   viz=False,  # optimize_on_joints
                   out_dir=None):
    """Run the fit for one specific image.
    :param targets: list of trimesh pcd
    :param j3ds: list of 3-D joints coordinate, N * 3 array
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """

    ###################################
    # 1. prior setting
    ###################################
    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))
    print(' gmm: w=', prior.weights.r)  # pose GMM statistics chumpy
    print(' gmm: m=', prior.means.shape)  # pose GMM statistics numpy
    print(" inital pose:", init_pose.reshape((-1, 3)))  # numpy

    ###################################
    # 3. fit
    ####################################

    choice = 'n'
    if os.path.isfile(os.path.join(out_dir, 'output_j3d.pkl')):
        choice = input("skip j3d? [y/n] ")

    if choice == 'n' or choice == 'N':

        svs = optimize_j3d_v2(
            j3ds,
            model,
            prior,  # priors
            n_betas=n_betas,  # shape params size
            targets=targets,
            viz=viz,  # visualizing or not
            regs=regs
        )

        params = {'svs': svs}

        with open(os.path.join(out_dir, 'output_j3d.pkl'), 'wb') as outf:  # 'wb' for python 3?
            pickle.dump(params, outf)
    else:
        with open(os.path.join(out_dir, 'output_j3d.pkl'), 'rb') as p:
            params = pickle.load(p)

    choice = 'n'
    if os.path.isfile(os.path.join(out_dir, 'output_icp.pkl')):
        choice = input("skip icp? [y/n] ")

    if choice == 'n' or choice == 'N':
        t0 = time()
        svs = optimize_icp_v2(
            targets,
            model,
            prior,  # priors
            n_betas=n_betas,  # shape params size
            maxiter=30,
            regs=regs,
            init=params,
            viz=viz,  # visualizing or not
            save_dir=out_dir
        )
        t1 = time()
        _LOGGER.info('elapsed %.05f', (t1 - t0))

        # 5. return resultant fit parameters  (pose, shape) is what we want but camera needed
        # save all Camera parameters and SMPL paramters
        params = {'svs': svs}

        with open(os.path.join(out_dir, 'output_icp.pkl'), 'wb') as outf:  # 'wb' for python 3?
            pickle.dump(params, outf)

    else:
        with open(os.path.join(out_dir, 'output_icp.pkl'), 'rb') as p:
            params = pickle.load(p)

    return params



    return params