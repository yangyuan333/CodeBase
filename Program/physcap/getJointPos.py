import pybullet_data
import pybullet as p
import math
import numpy as np

physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = r'E:\PanLiang_Programs\code\PhyCharacter\data\urdf/amass.urdf',
    # fileName = './physcap/physcap.urdf',
    # fileName = './Smpl2Urdf/yy1.urdf',
    basePosition = [1,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)

# for i in range(p.getNumJoints(physcapHuman)):
#     jointinfo = p.getJointInfo(physcapHuman, i)
#     if jointinfo[-1] == -1:
#         parentPos, _ = p.getBasePositionAndOrientation(physcapHuman)
#     else:
#         parentPos = p.getLinkState(physcapHuman, jointinfo[-1])[0]
#     sphereUid = p.createMultiBody(1, colSphereId, -1, np.array(jointinfo[-3])+np.array(parentPos), [0, 0, 0, 1])

for i in range(p.getNumJoints(physcapHuman)):
    sphereUid = p.createMultiBody(1, colSphereId, -1, p.getLinkState(physcapHuman,i)[4], [0, 0, 0, 1])

while(p.isConnected()):
    pass