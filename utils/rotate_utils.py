from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import torch
import pickle as pkl
import os
import sys
sys.path.append('./')
from utils.smpl_utils_YY import creatModel
from utils.obj_utils import read_obj, write_obj,MeshData

def Camera_project(points, externalMat=None, internalMat=None):
    '''
    points: n * 3
    externalMat: 4 * 4
    internalMat: 3 * 3
    return: ex_: n * 3
            in_: n * 2
    '''

    ## 类型判断
    if isinstance(points, list):
        points = np.array(points)

    if externalMat is None:
        return points
    else:
        if isinstance(externalMat, list):
            externalMat = np.array(externalMat)
        pointsInCamera = np.dot(externalMat, np.row_stack((points.T, np.ones(points.__len__()))))
        if internalMat is None:
            return pointsInCamera[:3,:].T
        else:
            if isinstance(internalMat, list):
                internalMat = np.array(internalMat)
            pointsInImage = np.dot(internalMat, pointsInCamera[:3,:]) / pointsInCamera[2,:][None,:]
            return pointsInImage[:2,:].T

def GetRotFromVecs(vec1, vec2):
    '''
    计算从vec1旋转到vec2所需的旋转矩阵
    vec1:n
    vec2:n
    return:3*3 np.array
    '''
    if isinstance(vec1, list):
        vec1 = np.array(vec1)
    if isinstance(vec2, list):
        vec2 = np.array(vec2)
    rotaxis = np.cross(vec1,vec2)
    rotaxis = rotaxis / np.linalg.norm(rotaxis)
    sita = math.acos(np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    r = R.from_rotvec(rotaxis*sita)
    return np.array(r.as_matrix())

def planeFit(points):
    '''
    从points中拟合一个平面方程
    z = a[0]*x + a[1]*y + a[2]
    '''
    A11 = 0
    A12 = 0
    A13 = 0
    A21 = 0
    A22 = 0
    A23 = 0
    A31 = 0
    A32 = 0
    A33 = 0
    S0 = 0
    S1 = 0
    S2 = 0
    if isinstance(points, list):
        points = np.array(points)
    for i in range(points.shape[0]):
        A11 += (points[i][0]*points[i][0])
        A12 += (points[i][0]*points[i][1])
        A13 += (points[i][0])
        A21 += (points[i][0]*points[i][1])
        A22 += (points[i][1]*points[i][1])
        A23 += (points[i][1])
        A31 += (points[i][0])
        A32 += (points[i][1])
        A33 += (1)
        S0 += (points[i][0]*points[i][2])
        S1 += (points[i][1]*points[i][2])
        S2 += (points[i][2])

    a = np.linalg.solve([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]],[S0,S1,S2])
    return a

def readVclCamparams(path):
    camIns = []
    camExs = []
    with open(path, 'r') as f:
        line = f.readline()
        camIdx = 0
        while(line):
            if line == (str(camIdx)+'\n'):
                camIdx += 1
                camIn = []
                camEx = []
                for i in range(3):
                    line = f.readline().strip().split()
                    camIn.append([float(line[0]), float(line[1]), float(line[2])])
                camIns.append(camIn)
                _ = f.readline()
                for i in range(3):
                    line = f.readline().strip().split()
                    camEx.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
                camEx.append([0.0, 0.0, 0.0, 1.0])
                camExs.append(camEx)
            line = f.readline()
    return camIns, camExs

def writeVclCamparams(path, camIns, camExs):
    with open(path, 'w') as f:
        camIdx = 0
        for camIn, camEx in zip(camIns, camExs):
            f.write(str(camIdx)+'\n')
            camIdx += 1
            for i in range(3):
                f.write(str(camIn[i][0]) + ' ' + str(camIn[i][1]) + ' ' + str(camIn[i][2]) + '\n')
            f.write('0 0\n')
            for i in range(3):
                f.write(str(camEx[i][0]) + ' ' + str(camEx[i][1]) + ' ' + str(camEx[i][2]) + ' ' + str(camEx[i][3]) + '\n')

def fitPlane(vertsPath, flag='z', ratio=100, **config):
    '''
    flag: which axis is up
    '''
    meshData = read_obj(vertsPath)
    joints = meshData.vert
    if flag == 'z':
        dimtrans = [0,1,2]
    elif flag == 'y':
        dimtrans = [0,2,1]
    elif flag == 'x':
        dimtrans = [1,2,0]
    jointsNew = np.array(joints)[:,dimtrans]
    a = planeFit(np.array(jointsNew))
    if 'savePath' in config:
        xyzMax = np.max(np.array(jointsNew), axis=0)
        xyzMin = np.min(np.array(jointsNew), axis=0)
        xyz = []
        for i in range(ratio):
            for j in range(ratio):
                x = xyzMin[0] + (xyzMax[0]-xyzMin[0]) * i / ratio
                y = xyzMin[1] + (xyzMax[1]-xyzMin[1]) * j / ratio
                z = a[0] * x + a[1] * y + a[2]
                xyz.append([x,y,z])
        meshData = MeshData()
        meshData.vert = np.array(xyz)[:,dimtrans]
        write_obj(config['savePath'], meshData) 
    return a

def rotateScene(scenePath, **config):
    meshData = read_obj(scenePath)
    if 'cam' in config:
        meshData.vert = Camera_project(np.array(meshData.vert),config['cam'])
    if 'savePath' in config:
        write_obj(config['savePath'], meshData)
    return meshData

def rotatePlane(a,pklPaths,temRotPaths,vn=[0.0,1.0,0.0],mode='smpl',gender='male',flag='z',**config):
    if mode.lower() == 'smpl':
        model = creatModel(gender=gender)
    elif mode.lower() == 'smplx':
        with open(pklPaths[0],'rb') as file:
            data = pkl.load(file)
        model = creatModel(
            mode,
            gender=gender,
            num_betas=data['person00']['betas'].shape[1],
            num_pca_comps=data['person00']['left_hand_pose'].shape[0])

    if flag == 'y':
        vec = np.array([
            a[0],
            -1.0,
            a[1]
        ])
    elif flag == 'z':
        vec = np.array([
            a[0],
            a[1],
            -1
        ])
    else:
        vec = np.array([
            -1,
            a[0],
            a[1]
        ])
    vn = np.array(vn)
    Rotm = GetRotFromVecs(vec, vn)

    if mode.lower() == 'smpl':
        for frameName, temPath in zip(pklPaths, temRotPaths):
            with open(frameName,'rb') as file:
                data = pkl.load(file)
            output = model(
                betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
                body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
                global_orient = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)),
                transl = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)))
            js = output.joints.detach().cpu().numpy().squeeze()
            j0 = js[0]
            data['person00']['pose'][:3] = (R.from_matrix(Rotm)*R.from_rotvec(data['person00']['pose'][:3])).as_rotvec()
            data['person00']['global_orient'] = data['person00']['pose'][:3]
            data['person00']['transl'] = R.from_matrix(Rotm).apply(j0 + data['person00']['transl']) - j0
            os.makedirs(os.path.dirname(temPath),exist_ok=True)
            with open(temPath,'wb') as file:
                pkl.dump(data, file)
    elif mode.lower() == 'smplx':
        for frameName, temPath in zip(pklPaths, temRotPaths):
            with open(frameName,'rb') as file:
                data = pkl.load(file)
            output = model(
                    betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
                    global_orient = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)),
                    body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
                    left_hand_pose = torch.tensor(data['person00']['left_hand_pose'][None,:].astype(np.float32)),
                    right_hand_pose = torch.tensor(data['person00']['right_hand_pose'][None,:].astype(np.float32)),
                    transl = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)),
                    jaw_pose = torch.tensor(data['person00']['jaw_pose'][None,:].astype(np.float32)))

            js = output.joints.detach().cpu().numpy().squeeze()
            j0 = js[0]
            data['person00']['global_orient'] = (R.from_matrix(Rotm)*R.from_rotvec(data['person00']['global_orient'])).as_rotvec()
            data['person00']['transl'] = R.from_matrix(Rotm).apply(j0 + data['person00']['transl']) - j0
            os.makedirs(os.path.dirname(temPath),exist_ok=True)
            with open(temPath,'wb') as file:
                pkl.dump(data, file)
    
    return Rotm

def transPlane(temRotPaths,savePaths,mode='smpl',gender='male',**config):
    if mode.lower() == 'smpl':
        model = creatModel(gender=gender)
    elif mode.lower() == 'smplx':
        with open(temRotPaths[0],'rb') as file:
            data = pkl.load(file)
        model = creatModel(
            mode,
            gender=gender,
            num_betas=data['person00']['betas'].shape[1],
            num_pca_comps=data['person00']['left_hand_pose'].shape[0])

    Js = []

    if mode.lower() == 'smpl':
        for framePath in temRotPaths:
            with open(framePath,'rb') as file:
                data = pkl.load(file)
            output = model(
                betas = torch.tensor(data['person00']['betas']),
                body_pose = torch.tensor(data['person00']['body_pose'][None,:]),
                global_orient = torch.tensor(data['person00']['global_orient'][None,:]),
                transl = torch.tensor(data['person00']['transl'][None,:]))
            js = output.joints.detach().cpu().numpy().squeeze()
        Js.append(js[10])
        Js.append(js[11])
    elif mode.lower() == 'smplx':
        for framePath in temRotPaths:
            with open(framePath,'rb') as file:
                data = pkl.load(file)
            output = model(
                    betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
                    global_orient = torch.tensor(data['person00']['global_orient'][None,:].astype(np.float32)),
                    body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
                    left_hand_pose = torch.tensor(data['person00']['left_hand_pose'][None,:].astype(np.float32)),
                    right_hand_pose = torch.tensor(data['person00']['right_hand_pose'][None,:].astype(np.float32)),
                    transl = torch.tensor(data['person00']['transl'][None,:].astype(np.float32)),
                    jaw_pose = torch.tensor(data['person00']['jaw_pose'][None,:].astype(np.float32)))
            js = output.joints.detach().cpu().numpy().squeeze()
            Js.append(js[10])
            Js.append(js[11])
    
    Js = np.array(Js)
    JsMean = np.mean(Js, axis = 0)

    for frameName, savePath in zip(temRotPaths, savePaths):
        with open(frameName,'rb') as file:
            data = pkl.load(file)
        data['person00']['transl'] -= JsMean
        os.makedirs(os.path.dirname(savePath),exist_ok=True)
        with open(savePath,'wb') as file:
            pkl.dump(data, file)
    
    return -JsMean

if __name__ == '__main__':
    pass