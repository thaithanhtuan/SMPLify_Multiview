'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This file defines the serialization functions of the SMPL model. 

Modules included:
- save_model:
  saves the SMPL model to a given file location as a .pkl file
- load_model:
  loads the SMPL model from a given file location (i.e. a .pkl file location), 
  or a dictionary object.

'''
from __future__ import print_function

__all__ = ['load_model', 'save_model']

import numpy as np

#import cPickle as pickle
import pickle
import chumpy as ch
from chumpy.ch import MatVecMult
from .posemapper import posemap  # relative path
from .verts import verts_core    # relative path

import scipy

"""

   model is saved in dictionary of the following keys

   v_template :  template vertices, Nv x 3
   J          :  maybe Joint location calculated by J_regressor
   weights    :  skin weight, Nv x Nj
   kintree    :  kinemetics tree 
   bs_type    :  ? male/female/neutral
   posedirs   :  pose directions angles (thetas) :  3 Jv  

   J_regressor:  vertices to Joint matrix, 3Nj x 3Nv
   J_regressor_prior: 
   Weight_prior: 
   shape_dir  :  shape PCA eginen vectors 
   - betas    :  it is not saved into SMPL model file itself, because it it specific to people.
   vert_sym_idxs : ?
   bs_style   : skinning method ('lbs')
   
   
   ## Posemap is not saved ^^;; why ?
  
   
"""
    
def save_model(model, fname):
    m0 = model
    trainer_dict = {'v_template': np.asarray(m0.v_template),'J': np.asarray(m0.J),'weights': np.asarray(m0.weights),'kintree_table': m0.kintree_table,'f': m0.f, 'bs_type': m0.bs_type, 'posedirs': np.asarray(m0.posedirs)}    
    if hasattr(model, 'J_regressor'):
        trainer_dict['J_regressor'] = m0.J_regressor
    if hasattr(model, 'J_regressor_prior'):
        trainer_dict['J_regressor_prior'] = m0.J_regressor_prior
    if hasattr(model, 'weights_prior'):
        trainer_dict['weights_prior'] = m0.weights_prior
    if hasattr(model, 'shapedirs'):
        trainer_dict['shapedirs'] = m0.shapedirs
    if hasattr(model, 'vert_sym_idxs'):
        trainer_dict['vert_sym_idxs'] = m0.vert_sym_idxs
    if hasattr(model, 'bs_style'):
        trainer_dict['bs_style'] = model.bs_style
    else:
        trainer_dict['bs_style'] = 'lbs'
    pickle.dump(trainer_dict, open(fname, 'w'), -1)


def backwards_compatibility_replacements(dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'


#
# re-construct all the require variables from the save pkl file 
#    some constant (learned) is used outside (Pmap)
#    some are used from pkl file (J_regression, Template, betas, pose_dir etc)
#    some are recalcauted  (Vertices, Joints)
#      
def ready_arguments(fname_or_dict):

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'),  encoding="latin1") # Python3 pickle issue
        #dd = pickle.load(open(fname_or_dict))
        
    else:
        dd = fname_or_dict   
        
    backwards_compatibility_replacements(dd)
        
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

            
    # eqn 6)  Tp(betas, thetas) =  T +  Bs(betas) + Bp(betas, thetas)        
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template'] # the vertices for specific betas (T +  Bs(betas))
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:,0])              # different from J, 
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:,1])              # J = J_regressor * V 
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:,2])              # so shape-dependent 
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T    
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose'])) # + Bp(betas, thetas)  
    else:   # when default shape used   
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose'])) # thetas *  (Pn * (Rn - Rn*))
            
    return dd


#
# main global function 
#
#  eqn-5  <=  eqn-6  <=   eqn-8, eqn-9, eqn- 10 
def load_model(fname_or_dict):
  
    dd = ready_arguments(fname_or_dict)         # load params,  eqn 6. using 8(betas), 9(pose), 10 (J) 
    
    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style']
    }
    
    result, Jtr = verts_core(**args)                        # vertices coords by Skin Blending (eq. 5)  
    result = result + dd['trans'].reshape((1,3))            # verices + global translation 
    result.J_transformed = Jtr + dd['trans'].reshape((1,3)) # Joints  + global translation

    for k, v in dd.items():
        print('-', k, 'type:', type(v), end='')
        if isinstance(v, str):
            print(v)
        elif isinstance(v, np.ndarray):
            print('dim:', v.shape)           
        elif isinstance(v, dict):
            print('keys:',v.keys())           
        elif isinstance(v, ch.ch.Ch):
           print('dim:', v.shape)                
        elif isinstance(v, ch.ch_ops.add):
            print(dir(v))
        elif isinstance(v, ch.reordering.transpose):
            print(dir(v))
        elif isinstance(v, scipy.sparse.csc.csc_matrix):
            print('dim:', v.shape)  
        else:
            print()
            
        setattr(result, k, v)                                # adding other params
        
        
    return result

