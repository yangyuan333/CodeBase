import pybullet_data
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import time
import argparse
import math
import sys
sys.path.append('./')
import Program.physcap.smplInPhyscap as smplInPhyscap
from utils.rotate_utils import *
import os
import pickle
from utils import rotate_utils, smpl_utils, obj_utils
import torch
import numpy as np
smplModel = smpl_utils.SMPLModel()
meshData = obj_utils.read_obj('./data/smpl/template.obj')
temVs, temFs = meshData.vert, meshData.face
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

class Config(object):
    show_smpl = True
    show_phscap = True
    show_smpl_in_physcap = True

if __name__ == '__main__': 
    lenth = 5
    physcapResultPath = r'./data/Physcap/Human36/inputData/S9/Directions/PhyCap_q.npy'
    kinematicResultPath = r'./data/Physcap/Human36/inputData/S9/Directions/0000000000.txt'
    smplResultPath = r'./data/Physcap/Human36/3Djoints/S9/Directions'
    # smplResultPath = r'./data/Physcap/saveData/0034'
    physcapJointOrderPath = r'./Program/physcap/data/jointIds.npy'
    qs = data_loader(physcapResultPath)
    framesData = motion_data_getter(kinematicResultPath).T
    qss = []
    for i in range(qs.__len__()):
        if (i+1) % 8 == 0:
            qss.append(qs[i])
    humanoid_path='./data/urdf/physcap.urdf'
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
    if Config.show_phscap:
        id_robot = p.loadURDF(humanoid_path, [0, 1, 0], globalScaling=1, useFixedBase=True)
    if Config.show_smpl_in_physcap:
        id_robot1 = p.loadURDF(humanoid_path, [0, 1, 1], globalScaling=1, useFixedBase=True, flags = p.URDF_MAINTAIN_LINK_ORDER)
    
    jointIds_reordered = np.load(physcapJointOrderPath)

    uid = p.addUserDebugParameter('frame', 0, qss.__len__(), 0)
    frameIdxLast = -1

    smpl_visual_ids = []
    smpl_collision_ids = []
    smpls = []
    idxsmpl = 0
    data_path = smplResultPath
    while(p.isConnected()):
        frameIdex = str(int(p.readUserDebugParameter(uid))).zfill(lenth)
        if frameIdxLast == frameIdex:
            continue

        frameIdxLast = frameIdex

        if Config.show_phscap:
            data = qss[int(p.readUserDebugParameter(uid))]
            motion_update_specification(id_robot, jointIds_reordered, data[6:])
            r = Rot.from_euler('zyx', data[3:6])  # Rot.from_matrix()
            angle = r.as_euler('xyz') 
            p.resetBasePositionAndOrientation(id_robot, [data[0]+2.35437/1000, data[1]+237.806/1000, data[2]-26.4052/1000], p.getQuaternionFromEuler([angle[2], angle[1], angle[0]]))
        
        if Config.show_smpl_in_physcap:
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

        if Config.show_smpl:
            file_path = os.path.join(data_path,frameIdex+'.pkl')
            #file_path = os.path.join(data_path,frameIdex+'.pkl')
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            poses = data['person00']['pose'] # 72维
            trans = data['person00']['transl']
            #poses = data['person00']['pose']
            #trans = data['person00']['transl']

            v2, j2 = smplModel(betas=torch.tensor(np.zeros((1, 10)).astype(np.float32)), thetas=torch.tensor(np.array(poses)[None,:].astype(np.float32)), trans=torch.tensor(np.array(trans)[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
            j2 = j2.squeeze(0).numpy()
            v2 = v2.squeeze(0).numpy()
            meshData.vert = v2
            obj_utils.write_obj('./data/temdata'+ str(idxsmpl) +'.obj', meshData)

            smpl_visual_ids.append(p.createVisualShape(
                                                        shapeType=p.GEOM_MESH,
                                                        fileName='./data/temdata'+ str(idxsmpl) +'.obj',
                                                        meshScale = 1,
                                                        rgbaColor = [255,0,0]
                                                    ))

            smpl_collision_ids.append(p.createVisualShape(
                                                            shapeType=p.GEOM_MESH,
                                                            fileName='./data/temdata'+ str(idxsmpl) +'.obj',
                                                            meshScale = 1
                                                        ))
            idxsmpl += 1
            if smpls.__len__() > 0:
                p.removeBody(smpls[-1])
            smpls.append(p.createMultiBody(
                                            baseCollisionShapeIndex=smpl_collision_ids[-1],
                                            baseVisualShapeIndex=smpl_visual_ids[-1],
                                            basePosition=[0,0,0]
                                        ))