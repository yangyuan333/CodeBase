import sys
sys.path.append('./')
import os
import pickle
from utils import rotate_utils
from Program.Smpl2Urdf import smplInAmassDof
from utils import obj_utils
from utils import smpl_utils
import math
pklPath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\motionDataPKL\GPA0000003566.pkl'
with open(pklPath, 'rb') as file:
    pklData = pickle.load(file)
print(1)

smplModel = smpl_utils.SMPLModel()
meshData = obj_utils.read_obj(r'./data/smpl/template.obj')

import pybullet as p
import pybullet_data
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
planeId = p.loadURDF(r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata/urdf/plane0000.urdf', [0,0,0], globalScaling=10, useMaximalCoordinates=True)
# amassDofHuman = p.loadURDF(
#     fileName = './data/urdf/amassdof.urdf',
#     basePosition = [0,1,0],
#     baseOrientation = p.getQuaternionFromEuler([0,0,0]),
#     globalScaling = 1,
#     useFixedBase = True,
#     flags = p.URDF_MAINTAIN_LINK_ORDER
# )

amassDofHuman = p.loadURDF(
    fileName = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata/urdf/amassdof.urdf',
)

poses = pklData['person00']['pose'] # 72
transl = pklData['person00']['transl'] # 3
betas = pklData['person00']['betas'] # 1 * 10
for key, value in enumerate(smplInAmassDof.smplInAmassDof.values()):
    pose = poses[key*3:(key*3+3)]
    if key == 0:
        p.resetBasePositionAndOrientation(amassDofHuman, transl, rotate_utils.R.from_rotvec(pose).as_quat())
        continue
    elur = rotate_utils.R.from_rotvec(pose).as_euler('XYZ', degrees=False)
    for rotKey, dim in enumerate(smplInAmassDof.smplInAmassDofDim[key*3:(key*3+3)]):
        if dim == 1:
            p.resetJointState(amassDofHuman, value, elur[rotKey])
        value+=1
import torch
import numpy as np
v2, j2 = smplModel(betas=torch.tensor(np.array(betas).astype(np.float32)), thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), trans=torch.tensor(np.array(transl)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
j2 = j2.squeeze(0).numpy()
v2 = v2.squeeze(0).numpy()
meshData.vert = v2
obj_utils.write_obj('./data/temdata'+ 'amass' +'.obj', meshData)

vs, js = smplModel(betas=torch.tensor(np.zeros((1,10)).astype(np.float32)), thetas=torch.tensor(np.zeros(72)[None,:].astype(np.float32)), trans=torch.tensor(np.zeros(3)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
js = js.squeeze(0).numpy()
vs = vs.squeeze(0).numpy()
print(js[0])
visulshape = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName='./data/temdata'+ 'amass' +'.obj',
    meshScale = 1,
    rgbaColor = [255,0,0]
    )

collisionshape = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName='./data/temdata'+ 'amass' +'.obj',
    meshScale = 1
    )

p.createMultiBody(
    baseCollisionShapeIndex=collisionshape,
    baseVisualShapeIndex=visulshape,
    basePosition=[0,0,0]
    )

while(p.isConnected()):
    continue