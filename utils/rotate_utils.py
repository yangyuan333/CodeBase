from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import sys
sys.path.append('./')
from utils.obj_utils import read_obj, write_obj, MeshData
from utils.smpl_utils import SMPLModel, smplxMain
import torch
import pickle as pkl
import smplx

def Camera_project(points, externalMat=None, internalMat=None):
    '''
    points: n * 3 np.array
    return: ex_: n * 3
            in_: n * 2
    '''
    if externalMat is None:
        return points
    else:
        pointsInCamera = np.dot(externalMat, np.row_stack((points.T, np.ones(points.__len__()))))
        if internalMat is None:
            return pointsInCamera[:3,:].T
        else:
            pointsInImage = np.dot(internalMat, pointsInCamera[:3,:]) / pointsInCamera[2,:][None,:]
            return pointsInImage[:2,:].T

def CalRotFromVecs(vec1, vec2):
    '''
    计算从vec1旋转到vec2所需的旋转矩阵
    '''
    rotaxis = np.cross(vec1,vec2)
    rotaxis = rotaxis / np.linalg.norm(rotaxis)
    sita = math.acos(np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    r = R.from_rotvec(rotaxis*sita)
    #rotvec = r.apply(vec1)
    return r

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

def fitPlane(config):
    meshData = read_obj(config['vertsPath'])
    joints = meshData.vert

    jointsNew = np.array(joints)[:,[0,2,1]]
    a = planeFit(np.array(jointsNew))
    xyzMax = np.max(np.array(jointsNew), axis=0)
    xyzMin = np.min(np.array(jointsNew), axis=0)
    xyz = []
    ratio = 100
    for i in range(ratio):
        for j in range(ratio):
            x = xyzMin[0] + (xyzMax[0]-xyzMin[0]) * i / ratio
            y = xyzMin[1] + (xyzMax[1]-xyzMin[1]) * j / ratio
            z = a[0] * x + a[1] * y + a[2]
            xyz.append([x,y,z])

    meshData = MeshData()
    meshData.vert = np.array(xyz)[:,[0,2,1]]
    write_obj(config['savePath'], meshData)
    
    return a

def rotateScene(config):
    meshData = read_obj(config['scenePath'])
    meshData.vert = Camera_project(np.array(meshData.vert),config['cam'])
    write_obj(config['savePath'], meshData)

def rotatePlane(config):
    if 'gender' not in config:
        Smpl = SMPLModel()
    elif config['gender'].lower() == 'male':
        Smpl = SMPLModel(model_path='./data/smpl/SMPL_MALE.pkl')
    elif config['gender'].lower() == 'female':
        Smpl = SMPLModel(model_path='./data/smpl/SMPL_FEMALE.pkl')
    elif config['gender'].lower() == 'neutral':
        Smpl = SMPLModel(model_path='./data/smpl/SMPL_NEUTRAL.pkl')
    vec = np.array([
        config['a'][0],
        -1.0,
        config['a'][1]
    ])
    vn = np.array(config['vn'])

    r = CalRotFromVecs(vec, vn)

    if ('model' in config) and (config['model'].lower() == 'smpl'):
        for frameName, temPath in zip(config['pklPaths'], config['temRotPaths']):
            with open(frameName,'rb') as file:
                data = pkl.load(file)
            pose = data['person00']['pose'].copy()
            betas = data['person00']['betas']
            transl = data['person00']['transl']
            pose[:3] *= 0
            _, js = Smpl(
                torch.tensor(betas.astype(np.float32)),
                torch.tensor(pose[None,:].astype(np.float32)),
                torch.tensor(np.array([[0,0,0]]).astype(np.float32)),
                torch.tensor([[1.0]])
            )
            j0 = js[0][0].numpy()
            data['person00']['pose'][:3] = (r*R.from_rotvec(data['person00']['pose'][:3])).as_rotvec()
            data['person00']['global_orient'] = data['person00']['pose'][:3]
            data['person00']['transl'] = r.apply(j0 + transl) - j0
            with open(temPath,'wb') as file:
                pkl.dump(data, file)
    elif ('model' in config) and (config['model'].lower() == 'smplx'):
        R'H:\YangYuan\Code\phy_program\CodeBase\data\models_smplx_v1_1\models',
        model = smplx.create(R'./data/models_smplx_v1_1/models', 'smplx',
                            gender=config['gender'], use_face_contour=False,
                            num_betas=config['num_betas'],
                            num_pca_comps=config['num_pca_comps'],
                            ext=config['ext'])
        for frameName, temPath in zip(config['pklPaths'], config['temRotPaths']):
            with open(frameName,'rb') as file:
                data = pkl.load(file)
            output = model(
                betas = torch.tensor(data['beta'][None,:]),
                global_orient = torch.tensor(data['global_orient']),
                body_pose = torch.tensor(data['body_pose']),
                left_hand_pose = torch.tensor(data['left_hand_pose']),
                right_hand_pose = torch.tensor(data['right_hand_pose']),
                transl = torch.tensor(data['transl']),
                jaw_pose = torch.tensor(data['jaw_pose']),
                return_verts = True,
            )
            pose = data['person00']['pose'].copy()
            betas = data['person00']['betas']
            transl = data['person00']['transl']
            pose[:3] *= 0
            _, js = Smpl(
                torch.tensor(betas.astype(np.float32)),
                torch.tensor(pose[None,:].astype(np.float32)),
                torch.tensor(np.array([[0,0,0]]).astype(np.float32)),
                torch.tensor([[1.0]])
            )
            j0 = js[0][0].numpy()
            data['person00']['pose'][:3] = (r*R.from_rotvec(data['person00']['pose'][:3])).as_rotvec()
            data['person00']['global_orient'] = data['person00']['pose'][:3]
            data['person00']['transl'] = r.apply(j0 + transl) - j0
            with open(temPath,'wb') as file:
                pkl.dump(data, file)  
        
    return r.as_matrix()

def transPlane(config):
    Smpl = SMPLModel()
    meshdata = read_obj(R"./data/smpl/template.obj")

    Js = []

    for framePath in config['temRotPaths']:
        with open(framePath, 'rb') as file:
            data = pkl.load(file)
        pose = data['person00']['pose']
        betas = data['person00']['betas']
        transl = data['person00']['transl']
        _, js = Smpl(
            torch.tensor(betas.astype(np.float32)),
            torch.tensor(pose[None,:].astype(np.float32)),
            torch.tensor(transl[None,:].astype(np.float32)),
            torch.tensor([[1.0]])
        )
        Js.append(js[0][10].numpy())
        Js.append(js[0][11].numpy())

    Js = np.array(Js)
    JsMean = np.mean(Js, axis = 0)

    for frameName, savePath in zip(config['temRotPaths'], config['savePaths']):
        with open(frameName,'rb') as file:
            data = pkl.load(file)
        data['person00']['transl'] -= JsMean
        with open(savePath,'wb') as file:
            pkl.dump(data, file)
    
    return -JsMean

def UnitestFitPlaneSmpl():
    config = {
        'vertsPath' : R'vs.txt',
        'savePath'  : R'plane.obj'
    }
    a = fitPlane(config)
    config = {
        'vn' : [0.0, 1.0, 0.0],
        'a'  : a,
        'temRotPaths' : [],
        'pklPaths'    : [],
    }

    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            config['pklPaths'].append(os.path.join(frameDir,'smpl','000_smpl_standard.pkl'))
            config['temRotPaths'].append(os.path.join(frameDir,'smpl','000_smpl_rot.pkl'))
    Rot = rotatePlane(config)
    print(Rot)
    config = {
        'temRotPaths' : [],
        'savePaths'    : [],
    }
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            config['temRotPaths'].append(os.path.join(frameDir,'smpl','000_smpl_rot.pkl'))
            config['savePaths'].append(os.path.join(frameDir,'smpl','000_smpl_final.pkl'))
    T = transPlane(config)
    print(T)

    config = {
        'scenePath' : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\scenes\vicon.obj',
        'savePath': R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\scenes\vicon_final.obj',
        'cam'     : np.vstack((np.hstack((Rot,T[:,None])),np.array([[0.0,0.0,0.0,1.0]])))
    }

    rotateScene(config)

if __name__ == '__main__':
    import glob
    import os
    import json
    path = r'E:\Evaluations_CVPR2022\Eval_Human36M10FPS'
    Human36 = ['S9', 'S11']
    for id in Human36:
        squenceIds = glob.glob(os.path.join(path,id,'camparams','*'))
        for squenceId in squenceIds:
            camPath = os.path.join(squenceId, 'camparams.txt')
            camIns, camExs = readVclCamparams(camPath)
            with open(os.path.join(squenceId, 'camIn0.json'), 'w') as f:
                f.write(json.dumps(camIns[0]))
            print(1)