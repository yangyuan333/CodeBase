import sys


sys.path.append('./')
from mesh_to_sdf import mesh_to_sdf
import trimesh
import torch
from utils.obj_utils import MeshData, read_obj, write_obj
from utils.rotate_utils import Camera_project, readVclCamparams
from utils.smpl_utils import pkl2smpl
import numpy as np
from itertools import product
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from scipy.spatial.transform import Rotation as R

## 测试smplx2smpl简易方法
if __name__ == '__main__':

    p = np.array([2,1])

    # with open(R'\\105.1.1.3\Hand\SkelEmbedding\BodyScan\exported_mesh2skel_smpl24_yangyuang\vicon_03301_01_s001.pkl','rb') as file:
    #     data1 = pkl.load(file)
    vs,js,fs = pkl2smpl(
        R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smplx\000_vcl.pkl',
        'smplx'
    )
    meshData = MeshData()
    img = cv2.imread(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\recordings\vicon_03301_01\img\s001_frame_00001__00.00.00.023.jpg')
    with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smplx\000_vcl.pkl','rb') as file:
        data = pkl.load(file)
    camIn = data['person00']['cam_intrinsic']
    camEx = data['person00']['cam_extrinsic']
    js = Camera_project(
        js, camEx, camIn
    )
    for j in js:
        img = cv2.circle(img,(int(j[0]),int(j[1])),2,(255,0,0))
        cv2.imshow('1',img)
        cv2.waitKey(0)

# if __name__ == '__main__':
#     a = torch.rand((3,3,3)).float()
#     b = F.grid_sample(
#         a.view(1,1,3,3,3),
#         torch.tensor([-1,-1,1]).float().view(1,1,1,1,3),
#         padding_mode='border'
#     )
#     print(b)
#     pass

'''
if __name__ == '__main__':

    ##  color test
    meshData = MeshData()
    meshData.vert = []
    meshData.color = []
    for i in range(10):
        for j in range(10):
            for k in range(10):
                meshData.vert.append([i*0.1,j*0.1,k*0.1])
                meshData.color.append([0,0,1])
    write_obj(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\scenes\color_test.obj',meshData)

    sdfData = {
        'min':None,
        'max':None,
        'dim':None,
        'data':None
    }

    meshData = read_obj(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\scenes\vicon_final.obj')
    verts, faces = np.array(meshData.vert), np.array(meshData.face)
    vmin = verts.min(0)
    vmax = verts.max(0)
    sdfData['min'] = vmin
    sdfData['max'] = vmax
    grid_dim = 256
    sdfData['dim'] = grid_dim
    mesh = trimesh.Trimesh(verts, faces-1, process=False)
    d1 = torch.linspace(vmin[0], vmax[0], grid_dim) # x
    d2 = torch.linspace(vmin[1], vmax[1], grid_dim) # y
    d3 = torch.linspace(vmin[2], vmax[2], grid_dim) # z
    meshx, meshy, meshz = torch.meshgrid((d1, d2, d3))
    qp = {
        (i,j,h): (meshx[i,j,h].item(), meshy[i,j,h].item(), meshz[i,j,h].item()) 
        for (i,j,h) in product(range(grid_dim), range(grid_dim), range(grid_dim))
    } ## h先变，j再变，i最后变 -- 先z轴遍历、再y轴遍历、最后x轴遍历
    qp_idxs = list(qp.keys())
    qp_values = np.array(list(qp.values()))

    qp_sdfs = mesh_to_sdf(mesh, qp_values, surface_point_method='sample')  # 10 secs
    qp_map = {qp_idxs[k]: qp_sdfs[k] for k in range(len(qp_sdfs))}
    qp_sdfs = np.zeros((grid_dim, grid_dim, grid_dim))

    for (i,j,h) in product(range(grid_dim), range(grid_dim), range(grid_dim)):
        qp_sdfs[i,j,h] = qp_map[(i,j,h)]
    sdfData['data'] = qp_sdfs
    # qp_sdfs = torch.tensor(qp_sdfs)

    with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\scenes\vicon_sdf.pkl','wb') as file:
        pkl.dump(sdfData,file)

    meshData = MeshData()
    meshData.vert = []
    meshData.color = []
    c1 = np.array([1,0,0])
    c2 = np.array([0,0,1])
    for (i,j,h) in product(range(grid_dim), range(grid_dim), range(grid_dim)):
        meshData.vert.append([meshx[i,j,h].item(), meshy[i,j,h].item(), meshz[i,j,h].item()])
        #c_d = c1 * qp_sdfs[i,j,h] if qp_sdfs[i,j,h] else c2 * qp_sdfs[i,j,h]
        c_d = c1 if qp_sdfs[i,j,h]>0 else c2
        meshData.color.append(c_d)
    write_obj(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\scenes\vicon_final_sdf.obj',meshData)




#     print('Generated grid sdf in {} secs'.format(time()-t))
'''