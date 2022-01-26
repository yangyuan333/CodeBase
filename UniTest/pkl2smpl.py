import pickle as pkl
import os
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel
from utils.obj_utils import write_obj, read_obj
from utils.rotate_utils import *
import torch
import numpy as np
import glob
import shutil

Smpl = SMPLModel()
meshdata = read_obj(R"./data/smpl/template.obj")

rootPath = R'C:\Users\yangyuan\Desktop\output (2)\ref_offset_adj_spline_fitting_sample\params'
saveRootPath = R'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\output\ref_offset_adj_spline_fitting_sample\pkl'

if os.path.exists(saveRootPath):
    shutil.rmtree(saveRootPath)
os.makedirs(saveRootPath, exist_ok=True)

dif = np.loadtxt('dif.txt')
exmat = np.array([
    [-0.65130947, 0.00445859, -0.75879911],
    [-0.00445859, -0.99998796, -0.00204879],
    [-0.75879911, 0.00204879, 0.65132151],
])
T = np.array([
    [18.08404511],
    [-1.70936021],
    [27.06832579]
])
idx = 0
for path in glob.glob(os.path.join(rootPath, '*')):
    pklPath = path
    savePath = saveRootPath
    with open(pklPath, 'rb') as file:
        pklData = pkl.load(file)
    if 'person00' in pklData:
        pose = pklData['person00']['pose']
        transl = pklData['person00']['transl']
        beta = pklData['person00']['betas']
    else:
        pose = pklData['pose']
        transl = pklData['transl']
        beta = pklData['betas']
    
    pose[0] = 0
    pose[1] = 0
    pose[2] = 0
    vs, js = Smpl(
        torch.tensor(beta.astype(np.float32)),
        torch.tensor(pose[None,:].astype(np.float32)),
        torch.tensor(np.array([0,0,0])[None,:].astype(np.float32)),
        torch.tensor([[1.0]])
    )

    pklData['person00']['pose'][:3] = (R.from_matrix(exmat)*R.from_rotvec(pklData['person00']['pose'][:3])).as_rotvec()
    pklData['transl'] = np.dot(exmat,js[0][0].numpy()[:,None])+np.dot(exmat,transl[:,None])+T-js[0][0].numpy()[:,None]-dif[idx][:,None]
    idx+=1

    fileName = os.path.basename(pklPath).split('.')[0]
    with open(os.path.join(savePath,fileName+".obj"),'wb') as file:
        pkl.dump(pklData, file)

rootPath = R'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\output\ref_offset_adj_spline_fitting_sample\pkl'
saveRootPath = R'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\output\ref_offset_adj_spline_fitting_sample\smpl1'

if os.path.exists(saveRootPath):
    shutil.rmtree(saveRootPath)
os.makedirs(saveRootPath, exist_ok=True)

for path in glob.glob(os.path.join(rootPath, '*')):
    pklPath = path
    savePath = saveRootPath
    with open(pklPath, 'rb') as file:
        pklData = pkl.load(file)
    if 'person00' in pklData:
        pose = pklData['person00']['pose']
        transl = pklData['person00']['transl']
        beta = pklData['person00']['betas']
    else:
        pose = pklData['pose']
        transl = pklData['transl']
        beta = pklData['betas']
    
    vs, js = Smpl(
        torch.tensor(beta.astype(np.float32)),
        torch.tensor(pose[None,:].astype(np.float32)),
        torch.tensor(transl[None,:].astype(np.float32)),
        torch.tensor([[1.0]])
    )
    meshdata.vert = vs[0].numpy()
    fileName = os.path.basename(pklPath).split('.')[0]
    write_obj(os.path.join(savePath,fileName+".obj"), meshdata)