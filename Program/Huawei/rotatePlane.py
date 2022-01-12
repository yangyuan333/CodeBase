import glob
import pickle as pkl
import numpy as np
import torch
import sys
import os
import shutil
sys.path.append('./')
from utils.smpl_utils import SMPLModel
from utils.obj_utils import read_obj, write_obj
from utils.rotate_utils import *
Smpl = SMPLModel()
meshData = read_obj(R'./data/smpl/template.obj')

a = [
    -0.02313439,
    0.0321033,
    1.02353208
]

vec = np.array([
    a[0],
    -1.0,
    a[1]
])
vn = np.array([
    0.0,
    1.0,
    0.0
])

r = CalRotFromVecs(vec, vn)

path = R'\\105.1.1.112\e\huawei_data\aligned'
video = 'demo3'
temPath = R'./data/temdata/huawei'
if os.path.exists(os.path.join(temPath,video)):
    shutil.rmtree(os.path.join(temPath,video))
os.makedirs(os.path.join(temPath,video),exist_ok=True)

framesName = glob.glob(os.path.join(path,video,'*'))
for frameName in framesName:
    with open(frameName,'rb') as file:
        data = pkl.load(file)
    pose = data['person00']['pose']
    Rnew = (r * R.from_rotvec(pose[:3])).as_rotvec()
    data['person00']['pose'][:3] = Rnew
    fileName = os.path.basename(frameName)
    with open(os.path.join(temPath,video,fileName),'wb') as file:
        pkl.dump(data, file)
