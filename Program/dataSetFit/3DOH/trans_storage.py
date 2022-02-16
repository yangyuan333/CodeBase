import glob
import os
import sys
sys.path.append('./')
import pickle as pkl
import shutil
import numpy as np
import torch
from utils.smpl_utils import SMPLModel,pkl2Smpl
smplModel = SMPLModel()
from utils.obj_utils import read_obj,write_obj
from utils.urdf_utils import smpl2Urdf
from utils.rotate_utils import readVclCamparams, Camera_project
meshData = read_obj(R'./data/smpl/template.obj')

def copyFiles(config, camIn, camEx):
    for path in glob.glob(os.path.join(config['rootPath'],'*')):
        with open(os.path.join(path,'000.pkl'),'rb') as file:
            data = pkl.load(file)
        dataSave = {}
        dataSave['person00'] = {}
        dataSave['person00']['betas'] = data['betas']
        dataSave['person00']['scale'] = np.array([1.0]).astype(np.float32)
        dataSave['person00']['global_orient'] = data['pose'][0][:3]
        dataSave['person00']['transl'] = data['transl'][0]
        dataSave['person00']['body_pose'] = data['pose'][0][3:]
        dataSave['person00']['pose'] = data['pose'][0]

        _, js = pkl2Smpl(os.path.join(path,'000.pkl'))
        dataSave['person00']['2D_joints'] = Camera_project(js,camEx,camIn)
        dataSave['person00']['cam_intrinsic'] = camIn
        dataSave['person00']['cam_extrinsic'] = camEx

        with open(os.path.join(config['savePath'],os.path.basename(path)+'.pkl'), 'wb') as file:
            pkl.dump(dataSave, file)

        # shutil.copyfile(
        #     os.path.join(path,'000.pkl'),
        #     os.path.join(config['savePath'],os.path.basename(path)+'.pkl')
        # )

def main(config):
    for dirs in glob.glob(os.path.join(config['rootPath'],'*')):
        fileName = os.path.basename(dirs)
        if os.path.exists(os.path.join(config['savePath'],fileName,'params')):
            shutil.rmtree(os.path.join(config['savePath'],fileName,'params'))
        os.makedirs(os.path.join(config['savePath'],fileName,'params'),exist_ok=True)
        configSeq = {}
        configSeq['rootPath'] = dirs
        configSeq['savePath'] = os.path.join(config['savePath'],fileName,'params')
        copyFiles(configSeq)
        configUrdf = {}
        configUrdf['pklPath'] = glob.glob(os.path.join(config['savePath'],fileName,'params','*'))[0]
        configUrdf['urdfPath'] = os.path.join(config['savePath'],fileName,'character.urdf')
        configUrdf['transPath'] = os.path.join(config['savePath'],fileName,'trans_offset.txt')
        configUrdf['isZero'] = True
        smpl2Urdf(configUrdf)

def main1(config):
    camIns, camExs = readVclCamparams(config['camPath'])
    for dirs in glob.glob(os.path.join(config['rootPath'],'*')):
        fileName = os.path.basename(dirs)
        if os.path.exists(os.path.join(config['savePath'],fileName,'params')):
            shutil.rmtree(os.path.join(config['savePath'],fileName,'params'))
        os.makedirs(os.path.join(config['savePath'],fileName,'params'),exist_ok=True)
        configSeq = {}
        configSeq['rootPath'] = dirs
        configSeq['savePath'] = os.path.join(config['savePath'],fileName,'params')
        copyFiles(configSeq, camIns[0], camExs[0])
        configUrdf = {}
        configUrdf['pklPath'] = glob.glob(os.path.join(config['savePath'],fileName,'params','*'))[0]
        configUrdf['urdfPath'] = os.path.join(config['savePath'],fileName,'character.urdf')
        configUrdf['transPath'] = os.path.join(config['savePath'],fileName,'trans_offset.txt')
        configUrdf['isZero'] = True
        smpl2Urdf(configUrdf)


if __name__ == '__main__':
    # config = {
    #     'rootPath':R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-GT\params_米制Y轴',
    #     'savePath':R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-physics'
    #     'camPath' :R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-GT\meter_camparams.txt'
    # }
    # main(config)
    config = {
        'rootPath':R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-GT\params_米制Y轴',
        'savePath':R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-physics',
        'camPath' :R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-GT\meter_camparams.txt'
    }
    main1(config)