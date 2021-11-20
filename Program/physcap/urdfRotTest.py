import pybullet as p
import pybullet_data
import math
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

physcapHuman = p.loadURDF(
    fileName = 'asset/physcap.urdf',
    basePosition = [0,1,0],
    baseOrientation = p.getQuaternionFromEuler([0,0,0]),
    globalScaling = 1,
    useFixedBase = True,
    flags = p.URDF_MAINTAIN_LINK_ORDER
)

print(p.getNumJoints(physcapHuman))

while(True):
    pass
