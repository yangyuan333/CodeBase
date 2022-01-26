import sys
sys.path.append('./')
from utils.rotate_utils import *
from utils.smpl_utils import SMPLModel,addRot
from utils.obj_utils import read_obj,write_obj,MeshData
import pickle as pkl
import torch
import numpy as np
import glob
import os

smplModel = SMPLModel()
meshData = read_obj('./data/smpl/template.obj')

mat = np.array([
    [-0.65130947, 0.00445859, -0.75879911],
    [-0.00445859, -0.99998796, -0.00204879],
    [-0.75879911, 0.00204879, 0.65132151]
])
T = np.array([18.08404511,-1.70936021,27.06832579])
difs = np.loadtxt('./dif.txt')
idx = -1
paths = R'C:\Users\yangyuan\Desktop\new_smooth'
savePath = R'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\output\new_smooth\pkl'
for path in glob.glob(os.path.join(paths, '*')):
    idx += 1
    with open(path, 'rb') as file:
        data = pkl.load(file)
    data['person00']['pose'],data['person00']['transl'],data['person00']['betas'] = addRot(data['person00']['pose'],data['person00']['transl'],data['person00']['betas'],mat,T-difs[idx])
    with open(os.path.join(savePath,os.path.basename(path)), 'wb') as file:
        pkl.dump(data, file)