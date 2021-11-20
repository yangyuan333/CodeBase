import pybullet as p
import pybullet_data
import sys
sys.path.append('./')
import os
import math

urdfPath = r'./Program/GPA/data/urdf/plane.urdf'
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF(urdfPath, [0,0,0], useMaximalCoordinates=True)

while(p.isConnected()):
    pass

