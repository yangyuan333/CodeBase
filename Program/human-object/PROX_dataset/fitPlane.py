import numpy as np
import os
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel,pkl2Smpl
from utils.obj_utils import MeshData, read_obj, write_obj
from utils.rotate_utils import *
import glob
import pickle as pkl
import torch

import shutil

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
    Smpl = SMPLModel()
    vec = np.array([
        config['a'][0],
        -1.0,
        config['a'][1]
    ])
    vn = np.array(config['vn'])

    r = CalRotFromVecs(vec, vn)

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

def checkSmpl(config):
    Smpl = SMPLModel()
    meshdata = read_obj(R"./data/smpl/template.obj")
    for pklPath, meshPath in zip(config['pklPaths'], config['meshPaths']):
        vs,_ = pkl2Smpl(pklPath)
        meshdata.vert = vs
        write_obj(meshPath, meshdata)

def rotateSmpl():
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
    Rot = rotatePlane(config)
    config = {
        'temRotPaths' : [],
        'savePaths'    : [],
    }
    T = transPlane(config)

if __name__ == '__main__':
    # config = {
    #     'vertsPath' : R'vs.txt',
    #     'savePath'  : R'plane.obj'
    # }
    # a = fitPlane(config)
    # config = {
    #     'vn' : [0.0, 1.0, 0.0],
    #     'a'  : a,
    #     'temRotPaths' : [],
    #     'pklPaths'    : [],
    # }

    # for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
    #     for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
    #         config['pklPaths'].append(os.path.join(frameDir,'smpl','000_smpl_standard.pkl'))
    #         config['temRotPaths'].append(os.path.join(frameDir,'smpl','000_smpl_rot.pkl'))
    # Rot = rotatePlane(config)
    # print(Rot)
    # config = {
    #     'temRotPaths' : [],
    #     'savePaths'    : [],
    # }
    # for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
    #     for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
    #         config['temRotPaths'].append(os.path.join(frameDir,'smpl','000_smpl_rot.pkl'))
    #         config['savePaths'].append(os.path.join(frameDir,'smpl','000_smpl_final.pkl'))
    # T = transPlane(config)
    # print(T)

    # config = {
    #     'pklPaths' : [],
    #     'meshPaths': [],
    # }
    # for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
    #     for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
    #         config['pklPaths'].append(os.path.join(frameDir,'smpl','000_smpl_final.pkl'))
    #         config['meshPaths'].append(os.path.join(frameDir,'smpl','000_smpl_final.obj'))
    # checkSmpl(config)

    # config = {
    #     'scenePath' : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\scenes\vicon.obj',
    #     'savePath': R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\scenes\vicon_final.obj',
    #     'cam'     : np.array(
    #         [
    #             [0.99947102, 0.02250866, 0.02347415, 1.340233],
    #             [-0.02250866, -0.04222833, 0.99885441, -0.00385183],
    #             [0.02347415, -0.99885441, -0.04169936, -0.8389669],
    #             [0.0, 0.0, 0.0, 1.0]
    #         ]
    #     ) 
    # }
    # rotateScene(config)

    # R1 = np.array([
    #         [0.99947102, 0.02250866, 0.02347415],
    #         [-0.02250866, -0.04222833, 0.99885441],
    #         [0.02347415, -0.99885441, -0.04169936]
    #     ])
    # T1 = np.array([
    #     [1.340233],
    #     [-0.00385183],
    #     [-0.8389669]
    # ])
    # Rcw = np.array([
    #     [0.5528847723992166, 0.0023366577652350353, -0.8332544440202931],
    #     [0.8331360441858384, 0.01553855830740956, 0.5528497852799715],
    #     [0.01423939350710162, -0.9998765389967993, 0.006644278466061525],
    # ])
    # Tcw = np.array([
    #     [2.1403999789236727],
    #     [-2.9695511941700334],
    #     [0.8682548100806671]
    # ])
    # Rote = np.linalg.inv(np.dot(R1,Rcw))
    # Te = -np.dot(np.dot(Rote,R1),Tcw) - np.dot(Rote,T1)
    # print(Rote)
    # print(Te)

    Cam = np.loadtxt(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\cam.txt')
    # meshData = read_obj(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smpl\000_smpl_final.obj')
    # meshData.vert = Camera_project(np.array(meshData.vert),Cam)
    # write_obj(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smpl\000_smpl_cam.obj',meshData)

    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            with open(os.path.join(frameDir,'smpl','000_smpl_final.pkl'), 'rb') as file:
                data = pkl.load(file)
            data['person00']['cam_extrinsic'] = Cam
            with open(os.path.join(frameDir,'smpl','000_smpl_final_cam.pkl'), 'wb') as file:
                pkl.dump(data,file)