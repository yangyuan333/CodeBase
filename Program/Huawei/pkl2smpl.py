import pickle as pkl
import os
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel
from utils.obj_utils import write_obj, read_obj
import torch
import numpy as np
import glob
import shutil

Smpl = SMPLModel()
meshdata = read_obj(R"./data/smpl/template.obj")

path = R'H:\YangYuan\Code\phy_program\CodeBase\data\temdata\results'
videos = ['demo1','demo2','demo3','params']
temPath = R'H:\YangYuan\Code\phy_program\CodeBase\data\temdata\results\trans'

for video in videos:
    if os.path.exists(os.path.join(temPath,video)):
        shutil.rmtree(os.path.join(temPath,video))
    os.makedirs(os.path.join(temPath,video),exist_ok=True)
    videoPath = os.path.join(path, video)
    videoPaths = glob.glob(os.path.join(videoPath,'*'))
    for framePath in videoPaths:
        with open(framePath, 'rb') as file:
            data = pkl.load(file)
        pose = data['person00']['pose']
        betas = data['person00']['betas']
        transl = data['person00']['transl']
        vs, js = Smpl(
            torch.tensor(betas.astype(np.float32)),
            torch.tensor(pose[None,:].astype(np.float32)),
            torch.tensor(transl[None,:].astype(np.float32)),
            torch.tensor([[1.0]])
        )
        meshdata.vert = vs[0].numpy()
        fileName = os.path.basename(framePath).split('.')[0]
        write_obj(os.path.join(temPath,video,fileName+".obj"), meshdata)