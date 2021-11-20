import pybullet as p
import pybullet_data
import math
import config, smplInPhyscap
import sys
sys.path.append('./')
from utils import rotate_utils, smpl_utils, obj_utils
import torch
import numpy as np

physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = 'physcap/physcap.urdf',
    basePosition = [0,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

f = open('./physcap/jointTrans.txt', 'w')
# for jointIdx in smplInPhyscap.smplInPhyscap.values():
#     if jointIdx == 0:
#         continue
#     jointInfo = p.getJointInfo(physcapHuman, jointIdx)[-3]
#     linkInfo = p.getLinkState(physcapHuman, p.getJointInfo(physcapHuman, jointIdx)[-1])[2]
#     f.write('1 0 0 ' + str(jointInfo[0]+linkInfo[0]) + '\n')
#     f.write('0 1 0 ' + str(jointInfo[1]+linkInfo[1]) + '\n')
#     f.write('0 0 1 ' + str(jointInfo[2]+linkInfo[2]) + '\n')
#     f.write('0 0 0 ' + '1' + '\n')
#     f.write('\n')
for jointIdx in range(p.getNumJoints(physcapHuman)):
    if jointIdx == 0:
        continue
    jointInfo = p.getJointInfo(physcapHuman, jointIdx)[-3]
    linkInfo = p.getLinkState(physcapHuman, p.getJointInfo(physcapHuman, jointIdx)[-1])[2]
    f.write('1 0 0 ' + str(jointInfo[0]+linkInfo[0]) + '\n')
    f.write('0 1 0 ' + str(jointInfo[1]+linkInfo[1]) + '\n')
    f.write('0 0 1 ' + str(jointInfo[2]+linkInfo[2]) + '\n')
    f.write('0 0 0 ' + '1' + '\n')
    f.write('\n')
# for i in range(1,100):
#     print(p.getJointInfo(physcapHuman, i)[-5])
#     print(p.getJointInfo(physcapHuman, i)[-3])
#     print(p.getLinkState(physcapHuman, p.getJointInfo(physcapHuman, i)[-1])[0])
#     print(p.getLinkState(physcapHuman, p.getJointInfo(physcapHuman, i)[-1])[2])
#     print(p.getLinkState(physcapHuman, p.getJointInfo(physcapHuman, i)[-1])[4])
#     print(' ')