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
    -0.03936695201524058,
    0.25952763576204146,
    1.1850749788447614
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
print(r.as_matrix())

path = R'./data/temdata/results'
video = 'demo3'
temPath = R'./data/temdata/results/huawei'
if os.path.exists(os.path.join(temPath,video)):
    shutil.rmtree(os.path.join(temPath,video))
os.makedirs(os.path.join(temPath,video),exist_ok=True)

framesName = glob.glob(os.path.join(path,video,'*'))
for frameName in framesName:
    with open(frameName,'rb') as file:
        data = pkl.load(file)
    pose = data['person00']['pose'].copy()
    betas = data['person00']['betas']
    transl = data['person00']['transl']
    pose[:3] *= 0
    _, js = Smpl(
        torch.tensor(betas.astype(np.float32)),
        torch.tensor(pose[None,:].astype(np.float32)),
        torch.tensor(np.array([[0,0,0]]).astype(np.float32)),
        torch.tensor([[1.0]])
    )
    j0 = js[0][0].numpy()
    data['person00']['pose'][:3] = (r*R.from_rotvec(data['person00']['pose'][:3])).as_rotvec()
    data['person00']['transl'] = r.apply(j0 + transl) - j0
    fileName = os.path.basename(frameName)
    with open(os.path.join(temPath,video,fileName),'wb') as file:
        pkl.dump(data, file)
