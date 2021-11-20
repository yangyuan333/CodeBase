import pybullet as p
import pybullet_data
import math
import sys
sys.path.append('./')
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_DOWN, 1)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

amassDofHuman = p.loadURDF(
    fileName = './data/temdata/amassOneDof.urdf',
    basePosition = [0,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

slider = []
jointIndex = []
for i in range(p.getNumJoints(amassDofHuman)):
    jointinfo = p.getJointInfo(amassDofHuman, i)
    joinName = jointinfo[1]
    print(i,' : ', str(joinName, 'utf-8'))
    if jointinfo[2] == p.JOINT_REVOLUTE:
        uid = p.addUserDebugParameter(str(joinName, 'utf-8'), -3.14, 3.14, 0)
        slider.append(uid)
        jointIndex.append(jointinfo[0])

while(p.isConnected()):
    for uid, index in zip(slider, jointIndex):
        value = p.readUserDebugParameter(uid)
        p.resetJointState(amassDofHuman, index, value)