import sys
sys.path.append('./')
from utils.rotate_utils import *
from utils.smpl_utils import SMPLModel
from utils.obj_utils import read_obj,write_obj,MeshData
import pickle as pkl
import torch
import numpy as np

smplModel = SMPLModel()
meshData = read_obj('./data/smpl/template.obj')

## 测试旋转参数
r = R.from_rotvec([0,3.14,0])
T = np.array([[10.4,2.5,6.4]])

## 读取一个smpl模型
smplPath = R'./UniTest/Smpl/data/origin.pkl'
with open(smplPath, 'rb') as file:
    data = pkl.load(file)
pose = data['person00']['pose'].copy()
betas = data['person00']['betas']
transl = data['person00']['transl']
vs,js = smplModel(
    torch.tensor(betas.astype(np.float32)),
    torch.tensor(pose[None,:].astype(np.float32)),
    torch.tensor(transl[None,:].astype(np.float32)),
    torch.tensor([[1.0]])
)
vs = vs[0].numpy()
js = js[0].numpy()
meshData.vert = vs
write_obj(R'./UniTest/Smpl/data/Smpl.obj',meshData)

## 旋转Smpl模型
vsR = r.apply(vs)
meshData.vert = vsR + T
write_obj(R'./UniTest/Smpl/data/rotSmpl.obj',meshData)

## 转换SMPL参数
pose[:3] *= 0
_,js = smplModel(
    torch.tensor(betas.astype(np.float32)),
    torch.tensor(pose[None,:].astype(np.float32)),
    torch.tensor(np.array([[0,0,0]]).astype(np.float32)),
    torch.tensor([[1.0]])
)
j0 = js[0][0].numpy()
data['person00']['pose'][:3] = (r*R.from_rotvec(data['person00']['pose'][:3])).as_rotvec()
data['person00']['transl'] = r.apply(j0 + transl) + T - j0
vs,_ = smplModel(
    torch.tensor(betas.astype(np.float32)),
    torch.tensor(data['person00']['pose'][None,:].astype(np.float32)),
    torch.tensor(data['person00']['transl'][None,:].astype(np.float32)),
    torch.tensor([[1.0]])
)
meshData.vert = vs[0].numpy()
write_obj(R'./UniTest/Smpl/data/rotPkl.obj',meshData)