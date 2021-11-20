import pybullet as p
import pybullet_data
import math
import config
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = './physcap/physcap.urdf',
    basePosition = [0,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

print(p.getNumJoints(physcapHuman))

slider = []

# for i in range(p.getNumJoints(physcapHuman)):
#     jointInfo = p.getJointInfo(physcapHuman, i)
#     uid = p.addUserDebugParameter(jointInfo[1].decode(), -3.14, 3.14, 0)
#     slider.append(uid)
import sys
sys.path.append('./')
import physcap.smplInPhyscap as smplInPhyscap
for key, value in smplInPhyscap.smplInPhyscap.items():
    uid = p.addUserDebugParameter(key, -3.14, 3.14, 0)
    slider.append(uid)

# for key, value in config.physcapHumanJointIdx.items():
#     uid = p.addUserDebugParameter(key, -3.14, 3.14, 0)
#     slider.append(uid)

# 动系欧拉
# while(p.isConnected()):
#     for joint_idex, jkey in enumerate(slider[1:]):
#         dof = p.readUserDebugParameter(jkey)
#         p.resetJointState(physcapHuman, joint_idex+1, dof)

while(p.isConnected()):
    for key, value in enumerate(smplInPhyscap.smplInPhyscap.values()):
        dof = p.readUserDebugParameter(slider[key])
        p.resetJointState(physcapHuman, value, dof)