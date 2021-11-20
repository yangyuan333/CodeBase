import pybullet_data
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import time
import argparse
import math
import sys
sys.path.append('./')
import physcap.smplInPhyscap as smplInPhyscap
from utils.rotate_utils import *

id_simulator = p.connect(p.GUI)
p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0)

humanoid_path='./physcap/physcap.urdf'
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
id_robot = p.loadURDF(humanoid_path, [0, 1, 0], globalScaling=1, useFixedBase=True)

import pickle
with open(r'E:\physcap_kinematic_3DOH\results\physcap/0013/00000.pkl', 'rb') as f:
    data = pickle.load(f)
print(1)