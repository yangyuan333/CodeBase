import pybullet as p
import pybullet_data
import math
import config
import numpy as np
import sys
sys.path.append('./')
import Program.physcap.smplInPhyscap as smplInPhyscap
from utils.rotate_utils import *
physicsCilent = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)

class Config(object):
    right_toe = 46 # 0.185
    right_heel = 47 # 0.194
    left_toe = 40 # 0.157
    left_heel = 41 # 0.164

    det = 0.000
    right_toe_avg = 0.02
    right_heel_avg = 0.02
    left_toe_avg = 0.01
    left_heel_avg = 0.01

physcapHuman = p.loadURDF(
    fileName = './data/urdf/physcap.urdf',
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

sample_contact_path = r'./data/Physcap/VCL/inputData/0029/sample_contact_0029.npy'
sample_stationary_path = r'./data/Physcap/VCL/inputData/0029/sample_stationary_0029.npy'
framesData = motion_data_getter(r'./data/Physcap/VCL/inputData/0029/0029.txt').T

sample_contact = []
filedata = []
for i in range(framesData.__len__()):
    data = framesData[i]
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
    contact = []
    if p.getLinkState(physcapHuman, Config.left_toe, 1)[0][1] < (Config.left_toe_avg+Config.det):
        contact.append(1)
    else:
        contact.append(0)
    if p.getLinkState(physcapHuman, Config.left_heel, 1)[0][1] < (Config.left_heel_avg+Config.det):
        contact.append(1)
    else:
        contact.append(0)
    if p.getLinkState(physcapHuman, Config.right_toe, 1)[0][1] < (Config.right_toe_avg+Config.det):
        contact.append(1)
    else:
        contact.append(0)
    if p.getLinkState(physcapHuman, Config.right_heel, 1)[0][1] < (Config.right_heel_avg+Config.det):
        contact.append(1)
    else:
        contact.append(0)
    sample_contact.append(contact)
np.save(sample_contact_path, sample_contact)
np.save(sample_stationary_path, np.ones(sample_contact.__len__()))