from ntpath import join
import pybullet as p
import pybullet_data
import math
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel
import torch
import numpy as np
from utils.obj_utils import read_obj, write_obj
import utils.urdf_utils as urdf_utils
from utils.rotate_utils import *
from utils import smpl_utils
import Program.Smpl2Urdf.config as config

smplModel = smpl_utils.SMPLModel()
meshData = read_obj('template.obj')
temVs, temFs = meshData.vert, meshData.face
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = r'./data/urdf/amass.urdf',
    # fileName = './physcap/physcap.urdf',
    # fileName = './data/temdata/yy1.urdf',
    basePosition = [0,0,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

data_path = R'H:\YangYuan\Code\phy_program\program_1\physcap\0013'
import os
import pickle
import time
uid = p.addUserDebugParameter('frame', 0, 30, 0)
idx = 0

smpl_visual_ids = []
smpl_collision_ids = []
smpls = []
idx = 0
frameIdxLast = -1
while(p.isConnected()):
    frameIdex = str(int(p.readUserDebugParameter(uid)+1)).zfill(5)
    if frameIdxLast == frameIdex:
        continue
    frameIdxLast = frameIdex
    file_path = os.path.join(data_path,frameIdex+'.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    poses = data['person00']['pose']
    trans = data['person00']['transl']
    betas = data['person00']['betas']
    # poses[0] = 0
    # poses[1] = 0
    # poses[2] = 0
    # trans = [0,0,0]
    # v2, j2 = smplModel(betas=torch.tensor(np.zeros((1, 10)).astype(np.float32)*5), thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), trans=torch.tensor(np.array(trans)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
    v2, j2 = smplModel(betas=torch.tensor(np.array(betas).astype(np.float32)), thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), trans=torch.tensor(np.array(trans)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
    j2 = j2.squeeze(0).numpy()
    v2 = v2.squeeze(0).numpy()
    # v2 = v2 - j2[0]
    meshData.vert = v2
    write_obj('./data/test'+ str(idx) +'.obj', meshData)

    smpl_visual_ids.append(p.createVisualShape(
                                                shapeType=p.GEOM_MESH,
                                                fileName='./data/test'+ str(idx) +'.obj',
                                                meshScale = 1,
                                                rgbaColor = [255,0,0]
                                            ))

    smpl_collision_ids.append(p.createVisualShape(
                                                    shapeType=p.GEOM_MESH,
                                                    fileName='./data/test'+ str(idx) +'.obj',
                                                    meshScale = 1
                                                ))
    idx += 1
    if smpls.__len__() > 0:
        p.removeBody(smpls[-1])
    smpls.append(p.createMultiBody(
                                    baseCollisionShapeIndex=smpl_collision_ids[-1],
                                    baseVisualShapeIndex=smpl_visual_ids[-1],
                                    basePosition=[0,0,0]
                                ))

    for i, key in enumerate(config.AmassInSmpl):
        pose = poses[key*3:(key*3+3)]
        p.resetJointStateMultiDof(physcapHuman, i, targetValue=R.from_rotvec(pose).as_quat())
    p.resetBasePositionAndOrientation(physcapHuman, trans, R.from_rotvec(poses[0:3]).as_quat())