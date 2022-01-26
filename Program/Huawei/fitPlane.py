import numpy as np
import os
import sys
sys.path.append('./')
from utils.obj_utils import read_obj, write_obj, MeshData
from utils.smpl_utils import SMPLModel
from utils.rotate_utils import planeFit
import glob
import pickle as pkl
import torch
Smpl = SMPLModel()

path = R'H:\YangYuan\Code\phy_program\CodeBase\data\temdata\results'
file = 'params'
joints = []

videoPath = os.path.join(path, file)
videoPaths = glob.glob(os.path.join(videoPath,'*'))
for framePath in videoPaths:
    with open(framePath, 'rb') as file:
        data = pkl.load(file)
    pose = data['person00']['pose']
    betas = data['person00']['betas']
    transl = data['person00']['transl']
    vs, js = Smpl(
        torch.tensor(betas.astype(np.float32)),
        torch.tensor(pose[None,:].astype(np.float32)),
        torch.tensor(transl[None,:].astype(np.float32)),
        torch.tensor([[1.0]])
    )
    joints.append(js[0][10].numpy())
    joints.append(js[0][11].numpy())

meshData = MeshData()
meshData.vert = joints
write_obj('./data/temdata/results/plane0.obj', meshData)

## 手动滤波
meshData = read_obj('./data/temdata/results/demo3Filter.obj')
joints = meshData.vert

jointsNew = np.array(joints)[:,[0,2,1]]
a = planeFit(np.array(jointsNew))
xyzMax = np.max(np.array(jointsNew), axis=0)
xyzMin = np.min(np.array(jointsNew), axis=0)
xyz = []
ratio = 100
for i in range(ratio):
    for j in range(ratio):
        x = xyzMin[0] + (xyzMax[0]-xyzMin[0]) * i / ratio
        y = xyzMin[1] + (xyzMax[1]-xyzMin[1]) * j / ratio
        z = a[0] * x + a[1] * y + a[2]
        xyz.append([x,y,z])

meshData = MeshData()
meshData.vert = np.array(xyz)[:,[0,2,1]]
write_obj('./data/temdata/results/plane1.obj', meshData)
