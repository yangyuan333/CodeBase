from scipy.spatial.transform import Rotation as R
import numpy as np
import glob
import os
import torch
import pickle as pkl
import sys
sys.path.append('./')
from utils.rotate_utils import fitPlane, rotatePlane, transPlane, rotateScene

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
            config['pklPaths'].append(os.path.join(frameDir,'000.pkl'))
            config['temRotPaths'].append(os.path.join(frameDir,'smplx','000_smplx_rot.pkl'))
    Rot = rotatePlane(config)
    print(Rot)
    config = {
        'temRotPaths' : [],
        'savePaths'    : [],
    }
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            config['temRotPaths'].append(os.path.join(frameDir,'smplx','000_smplx_rot.pkl'))
            config['savePaths'].append(os.path.join(frameDir,'smplx','000_smplx_final.pkl'))
    T = transPlane(config)
    print(T)

    config = {
        'scenePath' : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\scenes\vicon.obj',
        'savePath': R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\scenes\vicon_final.obj',
        'cam'     : np.vstack((np.hstack((Rot,T[:,None])),np.array([[0.0,0.0,0.0,1.0]])))
    }

    rotateScene(config)

def TransSmplx(pklPaths, savePaths):
    for pklPath, savePath in zip(pklPaths, savePaths):
        with open(pklPath, 'rb') as file:
            data = pkl.load(file, encoding='iso-8859-1')
            temData = {
                'person00' : {}
            }
            temData['person00']['betas'] = data['beta'][None,:]
            temData['person00']['body_pose'] = data['body_pose'][0]
            temData['person00']['jaw_pose'] = data['jaw_pose'][0]
            temData['person00']['right_hand_pose'] = data['right_hand_pose'][0]
            temData['person00']['left_hand_pose'] = data['left_hand_pose'][0]
            temData['person00']['global_orient'] = data['global_orient'][0]
            temData['person00']['transl'] = data['transl'][0]
            temData['person00']['num_pca_comps'] = data['num_pca_comps']
            with open(savePath, 'wb') as file:
                pkl.dump(temData, file)


if __name__ == '__main__':
    # pklPaths = []
    # savePaths = []
    # for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
    #     for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
    #         pklPaths.append(os.path.join(frameDir,'000.pkl'))
    #         savePaths.append(os.path.join(frameDir,'000n.pkl'))
    # TransSmplx(pklPaths, savePaths)
    
    # UnitestFitPlaneSmpl()
