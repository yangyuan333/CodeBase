import numpy as np
import os
import glob
import sys
import json
sys.path.append('./')
from utils import rotate_utils
from utils import obj_utils

Path = r'./Program/GTA/GTA-test'

points = np.loadtxt(os.path.join(Path, 'verts.txt'))
a = rotate_utils.planeFit(points)
vn_floor = np.array([a[0], a[1], -1])
vn = np.array([0,-1,0])
r = rotate_utils.CalRotFromVecs(vn_floor, vn)

scenceData = obj_utils.read_obj(os.path.join(Path, 'GTAtest.obj'))

scenceVsNew = np.array(r.apply(scenceData.vert)) - np.array(r.apply([0,0,a[2]]))

scenceMean = np.mean(scenceVsNew, axis=0)
scenceMean[1] = 0
scenceVsNew -= scenceMean

T = -np.array(r.apply([0,0,a[2]])) - scenceMean
# matrix = np.array(r.as_matrix())
# maex = np.column_stack((matrix, T[:,None]))
# np.savetxt('11111111.txt', maex)

keyspointsPath = glob.glob(os.path.join(r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTA-test\keypoints\TS1\Camera00', '*'))
savePath = r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTA-test\new\keypoints\TS1\Camera00'
os.makedirs(savePath, exist_ok=True)

for keyPath in keyspointsPath:
    with open(keyPath, 'rb') as file:
        keyData = json.load(file)
    joints3d = np.array(keyData['people'][0]['pose_keypoints_3d']).reshape(-1,4)
    joints3d[:,:3] = r.apply(joints3d[:,:3]) + T[None,:]
    keyData['people'][0]['pose_keypoints_3d'] = list(joints3d.reshape(-1))
    with open(os.path.join(savePath,os.path.basename(keyPath)), 'w') as file:
        json.dump(keyData, file)
# meshData = obj_utils.read_obj(r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTAIM-case1.obj')
# meshVsNew = np.array(r.apply(meshData.vert)) - np.array(r.apply([0,0,a[2]])) - scenceMean
# meshData.vert = list(meshVsNew)
# obj_utils.write_obj(os.path.join(Path, 'new', 'newScenceMesh.obj'), meshData)

# camPaths = glob.glob(os.path.join(Path, 'camparams', 'TS1', 'Camera00', '*'))
# savePath = os.path.join(Path, 'new', 'camparams', 'TS1', 'Camera00')
# os.makedirs(savePath, exist_ok=True)
# for camPath in camPaths[1:]:
#     camIns, camExs = rotate_utils.readVclCamparams(camPath)
#     camex = np.array(camExs[0])
#     camexnew = np.dot(camex[:3,:3], r.inv().as_matrix())
#     Tnew = camex[:3,3][:,None] - np.dot(camexnew, T[:,None])
#     camexnew = list(np.column_stack((camexnew, Tnew)))
#     rotate_utils.writeVclCamparams(os.path.join(savePath,os.path.basename(camPath)), camIns, [camexnew])