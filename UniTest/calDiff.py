import pickle as pkl
import os
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel
from utils.obj_utils import write_obj, read_obj, MeshData
from utils.rotate_utils import Camera_project
import torch
import numpy as np
import glob
import shutil
smplModel = SMPLModel()

pklPath1 = R'C:\Users\yangyuan\Desktop\new_smooth'
pklPath2 = R'C:\Users\yangyuan\Desktop\demo1'

ps = []
ps2=[]
for path in glob.glob(os.path.join(pklPath1,'*')):
    path2 = os.path.join(
        pklPath2, os.path.basename(path)
    )
    with open(path, 'rb') as file:
        data = pkl.load(file)
    vs, js = smplModel(
    torch.tensor(data['person00']['betas'].astype(np.float32)),
    torch.tensor(data['person00']['pose'][None,:].astype(np.float32)),
    torch.tensor(data['person00']['transl'][None,:].astype(np.float32)),
    torch.tensor([[1.0]])
    )
    exmat = np.linalg.inv(
        np.array([
        [-0.65130947, -0.0044586, -0.75879911,32.31011],
        [0.0044586, -0.99998796, 0.00204878,-1.8454262],
        [-0.75879911, -0.00204878, 0.65132151,-3.9115276],
        [0,0,0,1]
    ]))
    # exmat = np.linalg.inv(
    #     np.array([
    #     [-0.65130947, -0.0044586, -0.75879911,15.508025],
    #     [0.0044586, -0.99998796, 0.00204878,-0.24664348],
    #     [-0.75879911, -0.00204878, 0.65132151,-11.632322],
    #     [0,0,0,1]
    # ]))
    p = Camera_project(vs[0].numpy(),exmat)
    ps.append(p)

    with open(path2, 'rb') as file:
        data = pkl.load(file)
    vs, js = smplModel(
        torch.tensor(data['person00']['betas'].astype(np.float32)),
        torch.tensor(data['person00']['pose'][None,:].astype(np.float32)),
        torch.tensor(data['person00']['transl'][None,:].astype(np.float32)),
        torch.tensor([[1.0]])
    )
    exmat = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
    ])
    p1 = Camera_project(vs[0].numpy(),exmat)
    ps2.append(p1)

dif = np.array([0.0,0.0,0.0])
difs = []
for p1,p2 in zip(ps,ps2):
    dif += np.mean(p1-p2,axis=0)
    difs.append(np.mean(p1-p2,axis=0))
np.savetxt('dif.txt', np.array(difs))
print(dif/ps.__len__())

meshData = read_obj(R'./data/smpl/template.obj')
meshData.vert = p + np.array([[-17.34556369, 1.30602172, -7.94623741]])
write_obj(R'C:\Users\yangyuan\Desktop\test\1_2.obj',meshData)
