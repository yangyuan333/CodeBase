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
# 0.8
physcapHuman = p.loadURDF(
    fileName = r'./data/temdata/shape-ToeHeel.urdf',
    basePosition = [0,0.75,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)
print(p.URDF_MAINTAIN_LINK_ORDER)
# plane = p.loadURDF(
#     fileName = r'./data/temdata/shape-normalShape.urdf',
#     basePosition = [1,0.95,0],
#     baseOrientation = p.getQuaternionFromEuler([0,0,0]),
#     globalScaling = 1,
#     useFixedBase = True,
#     flags = p.URDF_MAINTAIN_LINK_ORDER
# )
print(p.getQuaternionFromEuler([0,0,0]))

for i in range(p.getNumJoints(physcapHuman)):
    jointInfo = p.getJointInfo(physcapHuman, i)
    print(jointInfo[1])

while(p.isConnected()):
    pass