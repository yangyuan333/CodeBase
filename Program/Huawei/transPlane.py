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

path = R'./data/temdata/huawei'
videos = ['demo3']
Js = []
for video in videos:
    videoPath = os.path.join(path, video)
    videoPaths = glob.glob(os.path.join(videoPath,'*'))
    for framePath in videoPaths:
        with open(framePath, 'rb') as file:
            data = pkl.load(file)
        pose = data['person00']['pose']
        betas = data['person00']['betas']
        transl = data['person00']['transl']
        vs, js = Smpl(
            torch.tensor(betas[None,:].astype(np.float32)),
            torch.tensor(pose[None,:].astype(np.float32)),
            torch.tensor(transl[None,:].astype(np.float32)),
            torch.tensor([[1.0]])
        )
        Js.append(js[0][10].numpy())
        Js.append(js[0][11].numpy())

Js = np.array(Js)
JsMean = np.mean(Js, axis = 0)
rotpath = R'./data/temdata/huawei'
transpath = R'./data/temdata/huaweiT'
if os.path.exists(os.path.join(transpath,videos[0])):
    shutil.rmtree(os.path.join(transpath,videos[0]))
os.makedirs(os.path.join(transpath,videos[0]),exist_ok=True)

framesName = glob.glob(os.path.join(rotpath,videos[0],'*'))
for frameName in framesName:
    with open(frameName,'rb') as file:
        data = pkl.load(file)
    data['person00']['transl'] -= JsMean
    fileName = os.path.basename(frameName)
    with open(os.path.join(transpath,videos[0],fileName),'wb') as file:
        pkl.dump(data, file)