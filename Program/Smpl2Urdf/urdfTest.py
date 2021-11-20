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

physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = r'E:\PanLiang_Programs\code\PhyCharacter\data\urdf/amass.urdf',
    #fileName = './physcap/physcap.urdf',
    #fileName = './Smpl2Urdf/yy1.urdf',
    basePosition = [1,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

physcapHuman1 = p.loadURDF(
    #fileName = r'E:\PanLiang_Programs\code\PhyCharacter\data\urdf/amass.urdf',
    #fileName = './physcap/physcap.urdf',
    fileName = './Smpl2Urdf/yy1.urdf',
    basePosition = [0,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

temVs, temFs = read_obj('./utils/template.obj')
smplModel = SMPLModel()
smpl_vs, smpl_js = smplModel(betas=torch.tensor(np.zeros((1, 10)).astype(np.float32)), thetas=torch.tensor(np.zeros((1, 72)).astype(np.float32)), trans=torch.tensor(np.zeros((1, 3)).astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
smpl_js = smpl_js.squeeze(0).numpy() # 24*3
smpl_vs = smpl_vs.squeeze(0).numpy()

rootpos = smpl_js[0]

colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[1], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[4], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[7], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[2], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[5], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[8], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[3], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[6], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[9], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[12], [0, 0, 0, 1])
sphereUid = p.createMultiBody(1, colSphereId, -1, np.array([0,1,0])-rootpos+smpl_js[15], [0, 0, 0, 1])
#sphereUid = p.createMultiBody(1, colSphereId, -1, [0,1,1], [0, 0, 0, 1])

while(p.isConnected()):
    pass