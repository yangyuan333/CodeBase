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

physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = r'./data/urdf/amass.urdf',
    #fileName = './physcap/physcap.urdf',
    #fileName = './Smpl2Urdf/yy.urdf',
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

sliders = []
for i in range(p.getNumJoints(physcapHuman)):
    jointslider = []
    for j in ['x','y','z']:
        uid = p.addUserDebugParameter(p.getJointInfo(physcapHuman, i)[1].decode()+j, -3.14, 3.14, 0)
        jointslider.append(uid)
    sliders.append(jointslider)

while(p.isConnected()):
    for joint_idex, jkey in enumerate(sliders):
        dofs = []
        for j in range(3):
            dofs.append(p.readUserDebugParameter(jkey[j]))
        p.resetJointStateMultiDof(physcapHuman, joint_idex, targetValue=R.from_euler('xyz',dofs).as_quat())
        p.resetJointStateMultiDof(physcapHuman1, joint_idex, targetValue=R.from_euler('xyz',dofs).as_quat())
        # p.resetJointState(physcapHuman, joint_idex+1, dof)