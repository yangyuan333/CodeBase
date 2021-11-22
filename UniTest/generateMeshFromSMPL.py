import os
import sys
sys.path.append('./')
from utils import smpl_utils, obj_utils
import pickle
import torch
import glob
smplModel = smpl_utils.SMPLModel()
meshData = obj_utils.read_obj(r'./data/smpl/template.obj')

path = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\pkl\0013_smooth_301_400'
savePath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\mesh\0013_smooth_301_400'
os.makedirs(savePath, exist_ok=True)
for pklPath in glob.glob(os.path.join(path, '*')):
    with open(pklPath, 'rb') as file:
        data = pickle.load(file)
    vs,js = smplModel(
        betas=torch.tensor(data['person00']['betas']), 
        thetas=torch.tensor(data['person00']['pose'][None,:]), 
        trans=torch.tensor(data['person00']['transl'][None,:]), 
        scale=torch.tensor(data['person00']['scale'][None,:])
    )
    meshData.vert = vs.squeeze(0).numpy()
    obj_utils.write_obj(os.path.join(savePath, os.path.basename(pklPath).split('.')[0]+'.obj'), meshData)