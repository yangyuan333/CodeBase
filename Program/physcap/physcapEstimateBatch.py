import pickle
import pybullet_data
import pybullet as p
import numpy as np
# from scipy.spatial.transform import Rotation as Rot
import time
import argparse
import math
import sys
sys.path.append('./')
import Program.physcap.smplInPhyscap as smplInPhyscap
from utils.rotate_utils import *
import os

id_simulator = p.connect(p.GUI)
p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0) 

def motion_update_specification(id_robot, jointIds, qs):
    [p.resetJointState(id_robot, jid, q) for jid, q in zip(jointIds, qs)]
    return 0

def visualizer(id_robot,q):
    motion_update_specification(id_robot, jointIds_reordered, q[6:]) 
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
    import glob
    rootPath = r'./data/Physcap/Human36'
    kinds = ['S9', 'S11']
    for kind in kinds:
        squenceIds = glob.glob(os.path.join(rootPath, 'inputData', kind, '*'))
        for squenceId in squenceIds:
            physcapResultPath = os.path.join(squenceId, 'PhyCap_q.npy')
            kinematicResultPath = os.path.join(squenceId, str(0).zfill(10)+'.txt')
            smplResultPath = os.path.join(rootPath, '3Djoints', kind, os.path.basename(squenceId))
            saveResultPath = os.path.join(rootPath, 'saveData', kind, os.path.basename(squenceId))
    
            physcapJointOrderPath = r'./Program/physcap/data/jointIds.npy'
            humanoid_path='./data/urdf/physcap.urdf'
            os.makedirs(saveResultPath, exist_ok=True)


            qs = data_loader(physcapResultPath)
            qss = []
            for i in range(qs.__len__()):
                if (i+1) % 8 == 0:
                    qss.append(qs[i])
            framesData = motion_data_getter(kinematicResultPath).T

            
            z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
            id_robot = p.loadURDF(humanoid_path, [0, 1, 0], globalScaling=1, useFixedBase=True)
            id_robot1 = p.loadURDF(humanoid_path, [0, 1, 1], globalScaling=1, useFixedBase=True, flags = p.URDF_MAINTAIN_LINK_ORDER)
            jointIds_reordered = np.load(physcapJointOrderPath)

            # print('\n')
            # for keyid, jid in enumerate(jointIds_reordered):
            #     jin = p.getJointInfo(id_robot, jid)
            #     print(keyid, ':', jin[1])

            robotDatas = []
            robot1Datas = []

            robotPoses = []
            robotTrans = []
            for i in range(qss.__len__()):
                data = qss[i]
                motion_update_specification(id_robot, jointIds_reordered, data[6:])
                r = R.from_euler('zyx', data[3:6])  # Rot.from_matrix()
                angle = r.as_euler('xyz')
                p.resetBasePositionAndOrientation(id_robot, [data[0]+2.35437/1000, data[1]+237.806/1000, data[2]-26.4052/1000], p.getQuaternionFromEuler([angle[2], angle[1], angle[0]]))

                robotTrans.append([data[0]+2.35437/1000, data[1]+237.806/1000, data[2]-26.4052/1000])

                robotPose = []
                robotPose += list(R.from_quat(p.getQuaternionFromEuler([angle[2], angle[1], angle[0]])).as_rotvec())
                for keyjoint, value in smplInPhyscap.smplInJointIds.items():
                    if value == -1:
                        robotPose += [0,0,0]
                        continue
                    dataindex = 0
                    euler = [0,0,0]
                    temdata = []
                    for dofidx in range(sum(smplInPhyscap.smplInJointIdsDof[keyjoint])):
                        temdata.append(data[6+value+dofidx])
                    for dofindex, dofbool in enumerate(smplInPhyscap.smplInJointIdsDof[keyjoint]):
                        if dofbool == 1:
                            euler[dofindex] = temdata[dataindex]
                            dataindex += 1
                    robotPose += list(R.from_euler('XYZ', euler).as_rotvec())
                robotPoses.append(robotPose)


                dataY = framesData[i]
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
                robotData = []
                robot1Data = []
                for i in smplInPhyscap.smplInPhyscapNoMain.values():
                    robotData.append(p.getLinkState(id_robot, i)[4])
                for i in smplInPhyscap.smplInPhyscap.values():
                    robot1Data.append(p.getLinkState(id_robot1, i)[4])
                robotDatas.append(robotData)
                robot1Datas.append(robot1Data)

            frameIdx = 0
            for datas, poses, trans in zip(robotDatas, robotPoses, robotTrans):
                joint_dict = {}
                frameJointData = []
                for data in datas:
                    frameJointData.append(data[0])
                    frameJointData.append(data[1])
                    frameJointData.append(data[2])
                joint_dict['joint3d'] = frameJointData
                joint_dict['person00'] = {}
                joint_dict['person00']['pose'] = poses
                joint_dict['person00']['transl'] = trans
                with open(os.path.join(saveResultPath,str(frameIdx).zfill(10)+'.pkl'), 'wb') as f:
                    pickle.dump(joint_dict, f)
                frameIdx += 1

            # file = open(r'E:\physcap_kinematic_3DOH\results/0013_Joint_physcap.txt', 'w')
            # for frame, framedata in enumerate(robotDatas):
            #     file.write(str(frame) + ' ')
            #     for data in framedata:
            #         file.write(str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ')
            #     file.write('\n')
            # file.close()
            # file = open(r'E:\physcap_kinematic_3DOH\results/0013_Joint.txt', 'w')
            # for frame, framedata in enumerate(robot1Datas):
            #     file.write(str(frame) + ' ')
            #     for data in framedata:
            #         file.write(str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ')
            #     file.write('\n')
            # file.close()
            # np.savetxt(r'E:\physcap_kinematic_3DOH\results/0029_Joint_physcap.txt', robotDatas)
            # np.savetxt(r'E:\physcap_kinematic_3DOH\results/0029_Joint.txt', robot1Datas)