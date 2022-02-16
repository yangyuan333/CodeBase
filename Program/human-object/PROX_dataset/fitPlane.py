import numpy as np
import os
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel
from utils.obj_utils import read_obj, write_obj
from utils.rotate_utils import *
import glob
import pickle as pkl
import torch

def fitPlane(config):
    Smpl = SMPLModel()
    meshData = read_obj(config['vertsPath'])
    joints = meshData.vert

    jointsNew = np.array(joints)[:,[0,2,1]]
    a = planeFit(np.array(jointsNew))
    xyzMax = np.max(np.array(jointsNew), axis=0)
    xyzMin = np.min(np.array(jointsNew), axis=0)
    xyz = []
    ratio = 100
    for i in range(ratio):
        for j in range(ratio):
            x = xyzMin[0] + (xyzMax[0]-xyzMin[0]) * i / ratio
            y = xyzMin[1] + (xyzMax[1]-xyzMin[1]) * j / ratio
            z = a[0] * x + a[1] * y + a[2]
            xyz.append([x,y,z])

    meshData = MeshData()
    meshData.vert = np.array(xyz)[:,[0,2,1]]
    write_obj(config['savePath'], meshData)
    
    return a

def rotatePlane(config):
    a = config['a']
    vec = np.array([
        a[0],
        -1.0,
        a[1]
    ])
    vn = np.array(config['vn'])

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


if __name__ == '__main__':
    config = {
        'vertsPath' : R'vs.txt',
        'savePath'  : R'plane.obj'
    }
    config = {
        'vn' : [0.0, 1.0, 0.0],
        'a'  : [],
    }