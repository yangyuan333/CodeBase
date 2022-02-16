import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import glob
import shutil
import pickle as pkl
import json
import torch
import sys
sys.path.append('./')
from utils.obj_utils import MeshData,read_obj,write_obj
from utils.rotate_utils import *
from utils.smpl_utils import SMPLModel, smplxMain

def ply2obj(config):
    plydata = PlyData.read(config['file_dir'])
    vs = plydata.elements[0].data
    fs = plydata.elements[1].data
    meshData = MeshData()
    meshData.vert = np.array(vs)
    meshData.face = [f[0]+1 for f in fs]
    write_obj(config['save_dir'], meshData)

def humanMeshTrans(path):
    config = {
        'file_dir':'',
        'save_dir':''
    }

    for dirPath in glob.glob(os.path.join(path, '*')):
        for framePath in glob.glob(os.path.join(dirPath,'meshes','*')):
            config['file_dir'] = glob.glob(os.path.join(framePath,'*'))[0]
            config['save_dir'] = os.path.join(framePath, os.path.basename(config['file_dir']).split('.')[0]+'.obj')
            ply2obj(config)

def fileCopy(path):
    config = {
        'file_dir':'',
        'save_dir':''
    }

    idx = 0
    for dirPath in glob.glob(os.path.join(path, '*')):
        for framePath in glob.glob(os.path.join(dirPath,'meshes','*')):
            config['file_dir'] = os.path.join(framePath, '000_world.obj')
            config['save_dir'] = os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\humanMesh\world', str(idx)+'.obj')
            idx += 1
            shutil.copyfile(config['file_dir'], config['save_dir'])

def cam2world(cam, path, savePath):
    meshData = read_obj(path)
    meshData.vert = Camera_project(np.array(meshData.vert), cam)
    write_obj(savePath, meshData)

def humanMesh2World(path):
    config = {
        'file_dir':'',
        'save_dir':''
    }

    cam = np.array([
        [0.5528847723992166, 0.0023366577652350353, -0.8332544440202931, 2.1403999789236727],
        [0.8331360441858384, 0.01553855830740956, 0.5528497852799715, -2.9695511941700334],
        [0.01423939350710162, -0.9998765389967993, 0.006644278466061525, 0.8682548100806671],
        [0.0, 0.0, 0.0, 1.0]
    ])

    for dirPath in glob.glob(os.path.join(path, '*')):
        for framePath in glob.glob(os.path.join(dirPath,'meshes','*')):
            config['file_dir'] = os.path.join(framePath, '000.obj')
            config['save_dir'] = os.path.join(framePath, '000_world.obj')
            cam2world(cam, config['file_dir'], config['save_dir'])

def smplxTest(config):
    return smplxMain(config)

if __name__ == '__main__':

    # humanMeshTrans(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh')
    
    # fileCopy(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh')

    # humanMesh2World(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh')

    config = {
        'modelPath' : R'H:\YangYuan\Code\phy_program\CodeBase\data\models_smplx_v1_1\models',
        'pklPath'   : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000.pkl',
        'savePath'  : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000x.obj',
        'gender'    : 'male',
        'num_betas' : 10,
        'num_pca_comps' : 12,
        'ext'       : 'npz',
    }
    
    vs, js = smplxTest(config)
