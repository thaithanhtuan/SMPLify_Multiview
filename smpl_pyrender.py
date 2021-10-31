###################################################################
# TO STUDY 
###################################################################
# 3D Triangle Mesh model 
# 3D mesh model files
# SMPL model 
# OpenGL 
# PyRender 
# BVH files 
# (c) 2020  heejune@seoultech.ac.kr
###################################################################


from smpl.serialization import load_model  
import numpy as np
#from bvh import Bvh, BvhNode
import math
import trimesh
import pyrender
import time
import pickle
    
def setupScene(cam_pose):
    
    """ Setup a scene, with a fixed camera pose  """ 

    # setup scene and camera  
    scene = pyrender.Scene( bg_color=np.array([0.0, 0.0, 0.0]),  
                            ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    cam   = pyrender.PerspectiveCamera(yfov=(np.pi* 75.0/ 180.0))     # 90 degree FOV
    cam_node = scene.add(cam, pose = cam_pose)
        
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True,  run_in_thread=True)  # a separate thread for rendering
    
    return scene, viewer
    

# global variables
smplMeshNode = None

def updateSMPL(scene, viewer, vertices, faces, color=None):

    global faceColors, smplMeshNode
    
    """ update the smpl mesh
        Currently, remove the existing one if any and add new one 
        TODO: can we just upate the vertices info keeping the mesh object is same?
        https://pyrender.readthedocs.io/en/latest/examples/viewer.html
    """ 
    # face-color used
    if color is None:
        color = np.ones([faces.shape[0], 4]) * [0.9, 0.5, 0.5, 1.0]     # pinky skin               
    
    triMesh = trimesh.Trimesh(vertices, faces, face_colors=color)        

    '''
    texture = None, f_colors = None, v_colors = None):     
        # 1. color setting     
        if texture is not None:
            raise Exception('not yet support texture (TODO)')
        elif f_colors is not None:
            triMesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors)
        elif v_colors is not None:
            v_colors = np.ones([vertices.shape[0], 4]) * [0.7, 0.7, 0.7, 0.8]   
            triMesh = trimesh.Trimesh(vertices, faces, vertex_colors=v_colors)
        else:
            f_colors = np.ones([faces.shape[0], 4]) * [0.9, 0.5, 0.5, 1.0]                  
            triMesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors)
    '''        
    # build a new mesh 
    mesh = pyrender.Mesh.from_trimesh(triMesh, smooth = False)   
    
    # update mesh 
    viewer.render_lock.acquire()
    if smplMeshNode is not None:
        scene.remove_node(smplMeshNode)
    # smplMeshNode = scene.add(mesh)
    viewer.render_lock.release()
    
    """  조인트 점찍는 방법 
    plot_joints = False  # TODO
    if plot_joints:
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)
    """
  

#
# update SMPL pose   
#
def poseSMPL(cam_pose, smpl, pose, viz = False):

    smpl.pose[:] = pose.flatten()
    
    if viz:
        visualize(cam_pose, smpl.r, smpl.f) 

    
'''
def runBVH(smpl, motion_file_path):

    
    with open(motion_file_path) as f:
            mocap = Bvh(f.read())
    
    joints = mocap.get_joints_names()
    print('# of joints:', len(joints), joints)
    print('# of frames:', mocap.nframes)
    allchannel = [0,1,2]  
    
    pose = np.zeros([24,3])

    #
    #           - 13 - 16 [lsh] - 18 [lelb] - 20 [lwr] - 22
    #       3 - 6 - 9 - 12 - 15
    #           - 14 - 17 [rsh] - 19 [relb] - 21 [rwr] - 23
    # 0 - 
    #       1 [lhip] - 4 [lknee] - 7 [lankle] - 10
    #       2 [rhip] - 5 [rknee] - 8 [rankle] - 11  
    #

    bvh2smpl_jtmap = {  'Hips': 0,
                    'Chest':6,
                    'Chest2':9,
                    'LeftCollar':13,
                    'LeftShoulder': 16,
                    'LeftElbow': 18,
                    'LeftWrist': 20,
                    'RightCollar':14,
                    'RightShoulder':17,
                    'RightElbow':19,
                    'RightWrist':21,
                    'Neck':12,
                    'Head':15, # not correct but..
                    'LeftHip':1,
                    'LeftKnee':4,
                    'LeftAnkle':7,
                    'RightHip':2,
                    'RightKnee':5, 
                    'RightAnkle':8 }

  
    cam_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],   # 90-rotation around z axis 
        [0.0,  1.0,  0.0,    0.0],   # 
        [0.0,  0.0,  1.0,   2.0],   # (0, 0, 0.0) translation  
        [0.0,  0.0,  0.0,   1.0]
    ])
  
    for  fn in range(mocap.nframes):
        print('fn:', fn);
        pose[:,:] = 0.0 # reset 
        for joint in joints:
            angs = mocap.frame_joint_channels(fn, joint, allchannel)
            #print('\t', joint, ':', math.radians(angs[0]), math.radians(angs[1]), math.radians(angs[2]) )
            smpljt = bvh2smpl_jtmap[joint]
            pose[smpljt,:] = (math.radians(angs[0]),math.radians(angs[1]), math.radians(angs[2]) )
            poseSMPL(cam_pose, smpl, pose, True)
'''   

if __name__ == "__main__":


    ## 1. Load SMPL model (here we load the male model)
    ## Make sure path is correct
    smpl_path = '/home/gu/workspace/Gu/SMPL/3.smpl_params_rendering/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    smpl = load_model(smpl_path)
    print("pose & beta size:", smpl.pose.size, ",", smpl.betas.size)
    print("pose :", smpl.pose)  # 24*3 = 72 (in 1-D)
    print("shape:", smpl.betas) # 10  (check the latent meaning..)
    print("vertex-x-range:", np.min(smpl.r[:,0]), np.max(smpl.r[:,0])) # x
    print("vertex-y-range:", np.min(smpl.r[:,1]), np.max(smpl.r[:,1])) # x
    print("vertex-z-range:", np.min(smpl.r[:,2]), np.max(smpl.r[:,2])) # x
    
    with open("/home/gu/workspace/Gu/SMPL/4.smplify/results/output20.pkl", 'rb') as p:
        smplify = pickle.load(p)
    
    smpl.pose[:] = smplify["pose"][:]
    smpl.betas[:] = smplify["betas"][:]
    
    print("my model")
    print("pose :", smpl.pose)  # 24*3 = 72 (in 1-D)
    print("shape:", smpl.betas) # 10  (check the latent meaning..)
    
    cam_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],   # zero-rotation  
        [0.0,  1.0,  0.0,  0.0],   # 
        [0.0,  0.0,  1.0,  2.0],   # (0, 0, 2.0) translation  
        [0.0,  0.0,  0.0,  1.0]
    ])
    scene, viewer = setupScene(cam_pose)

    objFilePath = "/home/gu/workspace/Gu/PIFu-master/results/pifu_demo/result_ryota.obj"
    target = trimesh.load(objFilePath)
    target_mesh = pyrender.Mesh.from_trimesh(target, smooth = False)

    updateSMPL(scene, viewer, smpl.r * smplify["scale"] + smplify["trans"], smpl.f)
    viewer.render_lock.acquire()
    scene.add(target_mesh)
    viewer.render_lock.release()
    
    x = input("finish?")    
       
    # Close the viwer (and its thread) to finish program     
    viewer.close_external()
    while viewer.is_active:
        pass
               
  
        