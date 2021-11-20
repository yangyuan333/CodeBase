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
meshData = read_obj('./data/smpl/template.obj')
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

plane52 = p.loadURDF(
    fileName = r'./data/urdf/plane0052.urdf',
    globalScaling = 10,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)
import pickle
data_path = R'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\motionDataPKL\side\GPA0000003566_SamCon.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)
poses = data['person00']['pose']
trans = data['person00']['transl']
betas = data['person00']['betas']
v2, j2 = smplModel(betas=torch.tensor(np.array(betas).astype(np.float32)), thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), trans=torch.tensor(np.array(trans)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
j2 = j2.squeeze(0).numpy()
v2 = v2.squeeze(0).numpy()
for i, key in enumerate(config.AmassInSmpl):
    pose = poses[key*3:(key*3+3)]
    p.resetJointStateMultiDof(physcapHuman, i, targetValue=R.from_rotvec(pose).as_quat())
p.resetBasePositionAndOrientation(physcapHuman, trans[0], R.from_rotvec(poses[0:3]).as_quat())

while(p.isConnected()):
    pass