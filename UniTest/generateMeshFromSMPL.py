import os
import sys
sys.path.append('./')
from utils import smpl_utils, obj_utils
import pickle
import torch
import glob
import shutil
import numpy as np
smplModel = smpl_utils.SMPLModel()
meshData = obj_utils.read_obj(r'./data/smpl/template.obj')

posePath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\pkl\0029_full_frc3_SMPL'
translPath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\pkl\0029_full_frc3_SMPL'
savePath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\mesh\0029_full_frc3_SMPL'
if os.path.exists(savePath):
    shutil.rmtree(savePath)
os.makedirs(savePath, exist_ok=True)
for pklPath in glob.glob(os.path.join(posePath, '*')):
    with open(pklPath, 'rb') as file:
        data = pickle.load(file)
    with open(os.path.join(translPath, os.path.basename(pklPath)), 'rb') as file:
        translData = pickle.load(file)
    vs,js = smplModel(
        betas=torch.tensor(data['person00']['betas'].astype(np.float32)), 
        thetas=torch.tensor(data['person00']['pose'][None,:].astype(np.float32)), 
        trans=torch.tensor(translData['person00']['transl'][None,:].astype(np.float32)), 
        scale=torch.tensor(np.array([[1.0]]).astype(np.float32))
    )
    meshData.vert = vs.squeeze(0).numpy()
    obj_utils.write_obj(os.path.join(savePath, os.path.basename(pklPath).split('.')[0]+'.obj'), meshData)

# path = r'\\105.1.1.112\Results_CVPR2022\kinematic-multiview-3DOH\results\0029'
# savePath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\mesh\0029'
# if os.path.exists(savePath):
#     shutil.rmtree(savePath)
# os.makedirs(savePath, exist_ok=True)
# for pklPath in glob.glob(os.path.join(path, '*')):
#     with open(pklPath, 'rb') as file:
#         data = pickle.load(file)
#     vs,js = smplModel(
#         betas=torch.tensor(data['person00']['betas']), 
#         thetas=torch.tensor(data['person00']['pose'][None,:]), 
#         trans=torch.tensor(data['person00']['transl'][None,:]), 
#         scale=torch.tensor(data['person00']['scale'][None,:])
#     )
#     meshData.vert = vs.squeeze(0).numpy()
#     obj_utils.write_obj(os.path.join(savePath, os.path.basename(pklPath).split('.')[0]+'.obj'), meshData)