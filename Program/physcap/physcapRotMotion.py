import pybullet as p
import pybullet_data
import math
import config
import numpy as np
import sys
sys.path.append('./')
import physcap.smplInPhyscap as smplInPhyscap
from utils.rotate_utils import *
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

def motion_data_getter(motion_filename):
    with open(motion_filename) as f:
        content = f.readlines() 
    content = np.array([x.strip().split(" ") for x in content])[1:]

    cleaned_pose = []

    for line in content:
        test = np.array([float(x) for x in line if not x == ""])[1:]
        cleaned_pose.append(test.tolist())
    cleaned_pose = np.array(cleaned_pose).T

    return cleaned_pose

framesData = motion_data_getter(r'E:\physcap_kinematic_3DOH\results/0029.txt').T

uid = p.addUserDebugParameter('frame', 0, framesData.__len__(), 0)
frameIdxLast = -1
while(p.isConnected()):
    # p.removeBody(smpl1)
    frameIdex = str(int(p.readUserDebugParameter(uid))).zfill(5)
    #frameIdex = str(int(p.readUserDebugParameter(uid)+1)).zfill(5)
    if frameIdxLast == frameIdex:
        continue
    frameIdxLast = frameIdex
    data = framesData[int(p.readUserDebugParameter(uid))]
    iidx = 6
    for key, indx in enumerate(smplInPhyscap.PhyscapDofInUrdf):
        idx = smplInPhyscap.PhyscapDofInUrdf[indx]
        dofs = smplInPhyscap.PhyscapSkeletonInSmplDim[indx]
        if key == 0:
            p.resetBasePositionAndOrientation(physcapHuman, data[0:3], R.from_euler('XYZ', data[3:6], degrees=False).as_quat())
            continue
        for keyk, dof in enumerate(dofs):
            if dof:
                p.resetJointState(physcapHuman, idx+keyk, data[iidx])
                iidx+=1