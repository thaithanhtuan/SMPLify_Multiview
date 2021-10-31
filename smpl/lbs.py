'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This file defines linear blend skinning for the SMPL loader which 
defines the effect of bones and blendshapes on the vertices of the template mesh.

Modules included:
- global_rigid_transformation: 
  computes global rotation & translation of the model
- verts_core: [overloaded function inherited from verts.verts_core]
  computes the blending of joint-influences for each vertex based on type of skinning

'''

from .posemapper import posemap  #relative path
#from .posemapper import posemap  #relative path
import chumpy
import numpy as np


# implementation eqn-3,-4  
 
def global_rigid_transformation(pose, J, kintree_table, xp):
    
    results = {}
    pose = pose.reshape((-1,3))  # 23x3 vectors to 69, if root angle included, 72  
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}
    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}  # A(k) in eqn-4

    #print(id_to_col)
    #print('kintree:', parent)
    
    if xp == chumpy:
        from .posemapper import Rodrigues
        rodrigues = lambda x : Rodrigues(x)
    else:
        import cv2
        rodrigues = lambda x : cv2.Rodrigues(x)[0]

    with_zeros = lambda x : xp.vstack((x, xp.array([[0.0, 0.0, 0.0, 1.0]])))   # cardtesian to homo
    results[0] = with_zeros(xp.hstack((rodrigues(pose[0,:]), J[0,:].reshape((3,1)))))        
        
    for i in range(1, kintree_table.shape[1]):  # Product of A(k)'s Rodrigues 
        results[i] = results[parent[i]].dot(with_zeros(xp.hstack((
            rodrigues(pose[i,:]),
            ((J[i,:] - J[parent[i],:]).reshape((3,1)))
            ))))

    pack = lambda x : xp.hstack([np.zeros((4, 3)), x.reshape((4,1))])
    
    results = [results[i] for i in sorted(results.keys())]
    results_global = results

    if True:
        
        results2 = [results[i] - (pack(
            results[i].dot(xp.concatenate( ( (J[i,:]), [0] ) )))  ## [0], not 0
            ) for i in range(len(results))]
        '''
        print('J.shape:', J.shape)
        print('len(results):', len(results))
        print('len(results[0]):', len(results[0]))
        print('len(results[0][0]):', len(results[0][0]))
        #print('results:', results)
        
        results2  = [ [] ]*len(results)
        print('results2:', results2)
        for i in range(len(results)):
            
            t0 = (J[i,:])
            print('t0:', t0)
            t1 = xp.concatenate( ( t0, [0] ) )
            print('t1:', t1)
            t2 = results[i].dot(t1)
            t3 = pack(t2) 
            results2[i] = results[i] - t3 
        
        '''
        results = results2
        
    result = xp.dstack(results)
    return result, results_global

 
# 
# LBS implementation for eqn-2, also same as eqn 5 = eqn 7 = eqn 11 = eqn 12    
#
def verts_core(pose, v, J, weights, kintree_table, want_Jtr=False, xp=chumpy):
 
    A, A_global = global_rigid_transformation(pose, J, kintree_table, xp)
    T = A.dot(weights.T)   # eqn-2's  W*G part

    '''
    print('weights: showing only ten from ', weights.shape[0])
    for i in range(10):
        idxes = np.nonzero(weights[i,:])
        print('at :',  idxes)
        print('val:',  weights[i, idxes])
    '''
    rest_shape_h = xp.vstack((v.T, np.ones((1, v.shape[0]))))  # cartesian to homogeneous coordinate
        
    v =(T[:,0,:] * rest_shape_h[0, :].reshape((1, -1)) +       # eqn-2's  multiply by T part 
        T[:,1,:] * rest_shape_h[1, :].reshape((1, -1)) + 
        T[:,2,:] * rest_shape_h[2, :].reshape((1, -1)) + 
        T[:,3,:] * rest_shape_h[3, :].reshape((1, -1))).T

    v = v[:,:3]             # back from homogeneous to cartesian coordinate 
    
    if not want_Jtr:        # 
        return v
    Jtr = xp.vstack([g[:3,3] for g in A_global])
    return (v, Jtr)
    
