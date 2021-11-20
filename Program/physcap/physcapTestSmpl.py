import pybullet as p
import pybullet_data
import math
import config, smplInPhyscap
import sys
sys.path.append('./')
from utils import rotate_utils, smpl_utils, obj_utils
import torch
import numpy as np
smplModel = smpl_utils.SMPLModel()
temVs, temFs = obj_utils.read_obj('template.obj')

physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
#z2y = p.getQuaternionFromEuler([0, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = 'physcap/physcap.urdf',
    basePosition = [0,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

print(p.getNumJoints(physcapHuman))

data_path = R'E:\Human-Training-v3.3\VCL Occlusion\params\0013'
#data_path = R'H:\YangYuan\Code\phy_program\program_1\physcap\0000'
import os
import pickle
import time
uid = p.addUserDebugParameter('frame', 0, 16, 0)
idx = 0

# smpl_visual_id = p.createVisualShape(
#                                         shapeType=p.GEOM_MESH,
#                                         fileName='./data/test.obj',
#                                         meshScale = 1,
#                                         rgbaColor = [255,0,0]
#                                     )
# smpl_collision_id = p.createVisualShape(
#                                             shapeType=p.GEOM_MESH,
#                                             fileName='./data/test.obj',
#                                             meshScale = 1
#                                         )
# smpl1 = p.createMultiBody(
#                             baseCollisionShapeIndex=smpl_collision_id,
#                             baseVisualShapeIndex=smpl_visual_id,
#                             basePosition=[0,1,0]
#                         )

smpl_visual_ids = []
smpl_collision_ids = []
smpls = []
idx = 0
frameIdxLast = -1
while(p.isConnected()):
    # p.removeBody(smpl1)
    frameIdex = str(int(p.readUserDebugParameter(uid))).zfill(5)
    #frameIdex = str(int(p.readUserDebugParameter(uid)+1)).zfill(5)
    if frameIdxLast == frameIdex:
        continue
    frameIdxLast = frameIdex
    file_path = os.path.join(data_path,frameIdex,'000.pkl')
    #file_path = os.path.join(data_path,frameIdex+'.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    poses = data['pose'][0] # 72维
    trans = data['transl']
    #poses = data['person00']['pose']
    #trans = data['person00']['transl']

    v2, j2 = smplModel(betas=torch.tensor(np.zeros((1, 10)).astype(np.float32)), thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), trans=torch.tensor(np.array(trans)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
    j2 = j2.squeeze(0).numpy()
    v2 = v2.squeeze(0).numpy()
    obj_utils.write_obj('./data/test'+ str(idx) +'.obj', v2, temFs)

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
        # smpl1 = p.createMultiBody(
        #                             baseCollisionShapeIndex=smpl_collision_id,
        #                             baseVisualShapeIndex=smpl_visual_id,
        #                             basePosition=[0,1,0]
        #                         )

    for key, value in enumerate(smplInPhyscap.smplInPhyscap.values()):
        pose = poses[key*3:(key*3+3)]
        if key == 0:
            #p.resetBasePositionAndOrientation(physcapHuman, [0,1,0], rotate_utils.R.from_rotvec([0,0,0]).as_quat())
            p.resetBasePositionAndOrientation(physcapHuman, trans, rotate_utils.R.from_rotvec(pose).as_quat())
        elur = rotate_utils.R.from_rotvec(pose).as_euler('XYZ', degrees=False)
        for rotKey, dim in enumerate(smplInPhyscap.smplInPhyscapDim[key*3:(key*3+3)]):
            if dim == 1:
                p.resetJointState(physcapHuman, value, elur[rotKey])
            value+=1

# while(p.isConnected()):
#     qKey = ord('q')
#     keys = p.getKeyboardEvents()
#     if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
#         # p.removeBody(smpl1)
#         frameIdex = str(int(p.readUserDebugParameter(uid))).zfill(5)
#         file_path = os.path.join(data_path,frameIdex,'000.pkl')
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#         poses = data['pose'][0] # 72维
#         trans = data['transl']

#         v2, j2 = smplModel(betas=torch.tensor(np.zeros((1, 10)).astype(np.float32)), thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), trans=torch.tensor(np.array(trans)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
#         j2 = j2.squeeze(0).numpy()
#         v2 = v2.squeeze(0).numpy()
#         obj_utils.write_obj('./data/test'+ str(idx) +'.obj', v2, temFs)

#         smpl_visual_ids.append(p.createVisualShape(
#                                                 shapeType=p.GEOM_MESH,
#                                                 fileName='./data/test'+ str(idx) +'.obj',
#                                                 meshScale = 1,
#                                                 rgbaColor = [255,0,0]
#                                             ))

#         # smpl_visual_id = p.createVisualShape(
#         #                                         shapeType=p.GEOM_MESH,
#         #                                         fileName='./data/test.obj',
#         #                                         meshScale = 1,
#         #                                         rgbaColor = [255,0,0]
#         #                                     )

#         smpl_collision_ids.append(p.createVisualShape(
#                                                     shapeType=p.GEOM_MESH,
#                                                     fileName='./data/test'+ str(idx) +'.obj',
#                                                     meshScale = 1
#                                                 ))
#         idx += 1
#         # smpl_collision_id = p.createVisualShape(
#         #                                             shapeType=p.GEOM_MESH,
#         #                                             fileName='./data/test.obj',
#         #                                             meshScale = 1
#         #                                         )
#         if smpls.__len__() > 0:
#             p.removeBody(smpls[-1])
#         smpls.append(p.createMultiBody(
#                                     baseCollisionShapeIndex=smpl_collision_ids[-1],
#                                     baseVisualShapeIndex=smpl_visual_ids[-1],
#                                     basePosition=[0,1,0]
#                                 ))
#         # smpl1 = p.createMultiBody(
#         #                             baseCollisionShapeIndex=smpl_collision_id,
#         #                             baseVisualShapeIndex=smpl_visual_id,
#         #                             basePosition=[0,1,0]
#         #                         )

#         for key, value in enumerate(smplInPhyscap.smplInPhyscap.values()):
#             pose = poses[key*3:(key*3+3)]
#             if key == 0:
#                 p.resetBasePositionAndOrientation(physcapHuman, [0,1,0], rotate_utils.R.from_rotvec([0,0,0]).as_quat())
#                 #p.resetBasePositionAndOrientation(physcapHuman, trans, rotate_utils.R.from_rotvec(pose).as_quat())
#             elur = rotate_utils.R.from_rotvec(pose).as_euler('XYZ', degrees=False)
#             for rotKey, dim in enumerate(smplInPhyscap.smplInPhyscapDim[key*3:(key*3+3)]):
#                 if dim == 1:
#                     p.resetJointState(physcapHuman, value, elur[rotKey])
#                 value+=1