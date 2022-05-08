import pybullet as p
import pybullet_data
import os
import numpy as np
import sys
sys.path.append('./')
import pickle
from Program.Smpl2Urdf import smplInAmassDof
from utils import rotate_utils
from utils import smpl_utils
from utils import obj_utils
import math
import pickle as pkl
with open(R'\\105.1.1.2\Body\Human-Data-Physics-v1.0\huawei_data\sihuawudao_smooth_norot\params\00409.pkl','rb') as file:
    data = pkl.load(file)
with open(R'C:\Users\yangyuan\Desktop\sihuawudaoV\params\00409.pkl','rb') as file:
    data1 = pkl.load(file)

smplModel = smpl_utils.SMPLModel()
##meshData = obj_utils.read_obj(r'./data/smpl/template.obj')
pklPath = R'C:\Users\yangyuan\Desktop\sihuawudaoV\params/00409.pkl'
with open(pklPath, 'rb') as file:
    pklData = pickle.load(file)
pose = pklData['person00']['pose']
transl = pklData['person00']['transl']

physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_DOWN, 1)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
amassDofHuman = p.loadURDF(
    fileName = R'C:\Users\yangyuan\Desktop\sihuawudaoV/test.urdf',
    basePosition = [0,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

for key, value in enumerate(smplInAmassDof.smplInAmassDofNew.values()):
    jointpose = pose[key*3:(key*3+3)]
    if key == 0:
        p.resetBasePositionAndOrientation(amassDofHuman, transl, rotate_utils.R.from_rotvec(jointpose).as_quat())
    if value == -1:
        continue
    jointpose = pose[key*3:(key*3+3)]
    euler = rotate_utils.R.from_rotvec(jointpose).as_euler('XYZ', degrees=False)
    for idx, dof in enumerate(smplInAmassDof.smplInAmassDofDimNew[key*3:(key*3+3)]):
        if dof == 1:
            p.resetJointState(amassDofHuman, value+idx, euler[idx])

while(True):
    pass

# import torch
# import numpy as np
# v2, j2 = smplModel(betas=torch.tensor(np.zeros((1,10)).astype(np.float32)), thetas=torch.tensor(np.array(pose)[None,:].astype(np.float32)), trans=torch.tensor(np.array(transl)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
# j2 = j2.squeeze(0).numpy()
# v2 = v2.squeeze(0).numpy()
# meshData.vert = v2
# obj_utils.write_obj('./data/temdata'+ 'amass' +'.obj', meshData)

# vs, js = smplModel(betas=torch.tensor(np.zeros((1,10)).astype(np.float32)), thetas=torch.tensor(np.zeros(72)[None,:].astype(np.float32)), trans=torch.tensor(np.zeros(3)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
# js = js.squeeze(0).numpy()
# vs = vs.squeeze(0).numpy()
# print(js[0])
# visulshape = p.createVisualShape(
#     shapeType=p.GEOM_MESH,
#     fileName='./data/temdata'+ 'amass' +'.obj',
#     meshScale = 1,
#     rgbaColor = [255,0,0]
#     )

# collisionshape = p.createVisualShape(
#     shapeType=p.GEOM_MESH,
#     fileName='./data/temdata'+ 'amass' +'.obj',
#     meshScale = 1
#     )

# p.createMultiBody(
#     baseCollisionShapeIndex=collisionshape,
#     baseVisualShapeIndex=visulshape,
#     basePosition=[0,0,0]
#     )

# while(p.isConnected()):
#     pass