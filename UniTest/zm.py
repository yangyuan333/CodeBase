import sys
sys.path.append('./')
import numpy as np
import glob
import os
from utils.smpl_utils import pkl2smpl
import smplx
import torch
import pickle as pkl
from utils.obj_utils import MeshData,write_obj
if __name__ == '__main__':
    saveSmplPath = R'\\105.1.1.2\Body\Human-Data-Physics-v2.0\GPA-testset\smpl'
    smplPklPath = R'\\105.1.1.2\Body\Human-Data-Physics-v2.0\GPA-testset\annotPkl'

    model = smplx.create('./data/smplData/body_models','smpl',gender='neutral')

    meshData = MeshData()

    for seq in glob.glob(os.path.join(smplPklPath,'*')):
        seqName = os.path.basename(seq)
        for camera in glob.glob(os.path.join(seq,'*')):
            cameraName = os.path.basename(camera)
            os.makedirs(os.path.join(saveSmplPath,seqName,cameraName),exist_ok=True)
            for frame in glob.glob(os.path.join(camera,'*')):
                savePath = os.path.join(
                    saveSmplPath,seqName,cameraName,os.path.basename(frame)[:-4]+'.obj')
                with open(frame,'rb') as file:
                    data = pkl.load(file)
                output = model(
                    betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
                    body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
                    global_orient = torch.tensor(data['person00']['global_orient'][None,:].astype(np.float32)),
                    transl = torch.tensor(data['person00']['transl'][None,:].astype(np.float32)))
                
                meshData.vert = output.vertices.detach().cpu().numpy().squeeze()
                meshData.face = model.faces + 1
                write_obj(savePath, meshData)