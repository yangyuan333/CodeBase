import sys
sys.path.append('./')
import os
import numpy as np
import pickle
import glob
import torch
from utils import smpl_utils
from utils import obj_utils

import pybullet as p
import pybullet_data
import sys
sys.path.append('./')
import os
import math

## pybullet
urdfPath = r'./Program/GPA/data/urdf/plane.urdf'
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF(urdfPath, [0,0,0], useMaximalCoordinates=True, globalScaling=10)

## load smpl model
smplModel = smpl_utils.SMPLModel()
meshData = obj_utils.read_obj('./data/smpl/template.obj')

## load frame data
path = r'E:\Results_CVPR2022\kinematic-multiview-GPA'
squencePath = os.path.join(path, '0000')

## 滑条
uid = p.addUserDebugParameter('frame', 0, glob.glob(os.path.join(squencePath,'*')).__len__(), 0)

smplId = -1
frameIdxLast = -1
while(p.isConnected()):
    frameIdex = str(int(p.readUserDebugParameter(uid))).zfill(10)
    if frameIdxLast == frameIdex:
        continue
    frameIdxLast = frameIdex
    file_path = os.path.join(squencePath, frameIdex+'.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    poses = data['person00']['pose']
    trans = data['person00']['transl']
    betas = data['person00']['betas']
    
    vs, js = smplModel(
        betas=torch.tensor(np.array(betas).astype(np.float32)), 
        thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), 
        trans=torch.tensor(np.array(trans)[None,:].astype(np.float32)), 
        scale=torch.tensor([1]), 
        gR=None, lsp=False)
    js = js.squeeze(0).numpy()
    vs = vs.squeeze(0).numpy()

    if os.path.exists(os.path.join('./data/temdata/tem.obj')):
        os.remove(os.path.join('./data/temdata/tem.obj'))
    meshData.vert = vs
    obj_utils.write_obj(os.path.join('./data/temdata/tem.obj'), meshData)

    visualshape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=os.path.join('./data/temdata/tem.obj'),
        meshScale = 1,
        rgbaColor = [255,0,0])
    collosionshape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=os.path.join('./data/temdata/tem.obj'),
        meshScale = 1)

    if smplId != -1:
        p.removeBody(smplId)
    smplId = p.createMultiBody(
        baseCollisionShapeIndex=visualshape,
        baseVisualShapeIndex=collosionshape,
        basePosition=[0,0,0]
        )

    # for i, key in enumerate(config.AmassInSmpl):
    #     pose = poses[key*3:(key*3+3)]
    #     p.resetJointStateMultiDof(physcapHuman, i, targetValue=R.from_rotvec(pose).as_quat())
    # p.resetBasePositionAndOrientation(physcapHuman, trans, R.from_rotvec(poses[0:3]).as_quat())