# import pybullet as p
# import pybullet_data
# import math
# import config
# import numpy as np
# import sys
# sys.path.append('./')
# import physcap.smplInPhyscap as smplInPhyscap
# from utils.rotate_utils import *
# physicsCilent = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
# z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
# z2y = p.getQuaternionFromEuler([0, 0, 0])
# planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
# physcapHuman = p.loadURDF(
#     fileName = './physcap/physcap.urdf',
#     basePosition = [0,1,0],
#     baseOrientation = p.getQuaternionFromEuler([0,0,0]),
#     globalScaling = 1,
#     useFixedBase = False,
#     flags = p.URDF_MAINTAIN_LINK_ORDER
# )
# def motion_update_specification(id_robot, jointIds, qs):
#     [p.resetJointState(id_robot, jid, q) for jid, q in zip(jointIds, qs)]
#     return 0
# jointIds = np.load('./physcap/jointIds.npy')
# motionData = np.load('./physcap/PhyCap_q.npy')
# uid = p.addUserDebugParameter('frame', 0, motionData.__len__(), 0)
# frameIdxLast = -1
# import time
# for data in motionData:
#     motion_update_specification(physcapHuman, jointIds, data[6:])
#     time.sleep(0.002)
# # while(p.isConnected()):
# #     frameIdex = str(int(p.readUserDebugParameter(uid)+1)).zfill(5)
# #     if frameIdxLast == frameIdex:
# #         continue
# #     frameIdxLast = frameIdex
# #     data = motionData[int(p.readUserDebugParameter(uid))]
# #     motion_update_specification(physcapHuman, jointIds, data[6:])
# #     r = R.from_euler('zyx', data[3:6])  # Rot.from_matrix()
# #     angle = r.as_euler('xyz')
# #     # p.resetBasePositionAndOrientation(physcapHuman, [data[0], data[1], data[2]], p.getQuaternionFromEuler([angle[2], angle[1], angle[0]]))

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

def motion_update_specification(id_robot, jointIds, qs):
    [p.resetJointState(id_robot, jid, q) for jid, q in zip(jointIds, qs)]
    return 0

def visualizer(id_robot,q):
    motion_update_specification(id_robot, jointIds_reordered, q[6:]) 
    # r = Rot.from_euler('zyx', q[3:6])  # Rot.from_matrix()
    # angle = r.as_euler('xyz') 
    # p.resetBasePositionAndOrientation(id_robot, [q[0], q[1], q[2]], p.getQuaternionFromEuler([angle[2], angle[1], angle[0]]))
    # p.stepSimulation()
    return 0
def data_loader(q_path):
    return np.load(q_path) 

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

if __name__ == '__main__': 
    qs = data_loader(r'E:\physcap_kinematic_3DOH\results/PhyCap_q_0029.npy')
    qss = []
    for i in range(qs.__len__()):
        if (i+1) % 8 == 0:
            qss.append(qs[i])
    framesData = motion_data_getter(r'E:\physcap_kinematic_3DOH\results/0029.txt').T
    humanoid_path='./physcap/physcap.urdf'
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
    id_robot = p.loadURDF(humanoid_path, [0, 1, 0], globalScaling=1, useFixedBase=True)
    id_robot1 = p.loadURDF(humanoid_path, [0, 1, 1], globalScaling=1, useFixedBase=True, flags = p.URDF_MAINTAIN_LINK_ORDER)
    
    # print('physcap:')
    # for i in range(p.getNumJoints(id_robot)):
    #     print(p.getJointInfo(id_robot, i)[1])
    # print('maintain:')
    # for i in range(p.getNumJoints(id_robot1)):
    #     print(p.getJointInfo(id_robot1, i)[1])
    # colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
    # for i in range(p.getNumJoints(id_robot)):
    #     if i == 0:
    #         continue
    #     jointinfo = p.getJointInfo(id_robot, i)
    #     linkstate = p.getLinkState(id_robot, jointinfo[-1])
    #     sphereUid = p.createMultiBody(1, colSphereId, -1, np.array(jointinfo[-3])+np.array(linkstate[0]), [0, 0, 0, 1])
    
    # colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
    # sphereUids = []
    # for i in smplInPhyscap.smplInPhyscapNoMain.values():
    #     jointinfo = p.getJointInfo(id_robot, i)
    #     sphereUid = p.createMultiBody(1, colSphereId, -1, p.getLinkState(id_robot,i)[4], [0, 0, 0, 1])
    #     sphereUids.append(sphereUid)

    jointIds_reordered = np.load('./physcap/jointIds.npy')

    uid = p.addUserDebugParameter('frame', 0, qss.__len__(), 0)
    frameIdxLast = -1

    while(p.isConnected()):
        frameIdex = str(int(p.readUserDebugParameter(uid))).zfill(5)
        if frameIdxLast == frameIdex:
            continue
        # for suid in sphereUids:
        #     p.removeBody(suid)
        # sphereUids = []
        frameIdxLast = frameIdex
        data = qss[int(p.readUserDebugParameter(uid))]
        motion_update_specification(id_robot, jointIds_reordered, data[6:])
        r = Rot.from_euler('zyx', data[3:6])  # Rot.from_matrix()
        angle = r.as_euler('xyz') 
        p.resetBasePositionAndOrientation(id_robot, [data[0]+2.35437/1000, data[1]+237.806/1000, data[2]-26.4052/1000], p.getQuaternionFromEuler([angle[2], angle[1], angle[0]]))

        dataY = framesData[int(p.readUserDebugParameter(uid))]
        iidx = 6
        for key, indx in enumerate(smplInPhyscap.PhyscapDofInUrdf):
            idx = smplInPhyscap.PhyscapDofInUrdf[indx]
            dofs = smplInPhyscap.PhyscapSkeletonInSmplDim[indx]
            if key == 0:
                p.resetBasePositionAndOrientation(id_robot1, dataY[0:3], R.from_euler('XYZ', dataY[3:6], degrees=False).as_quat())
                continue
            for keyk, dof in enumerate(dofs):
                if dof:
                    p.resetJointState(id_robot1, idx+keyk, dataY[iidx])
                    iidx+=1
        # for i in smplInPhyscap.smplInPhyscapNoMain.values():
        #     jointinfo = p.getJointInfo(id_robot, i)
        #     sphereUid = p.createMultiBody(1, colSphereId, -1, p.getLinkState(id_robot,i)[4], [0, 0, 0, 1])
        #     sphereUids.append(sphereUid)