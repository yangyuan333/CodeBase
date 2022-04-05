import sys
sys.path.append('./')
import numpy as np
import pickle as pkl
import cv2
import glob
import os
import json

from utils.obj_utils import MeshData, read_obj, write_obj
from utils.smpl_utils import pkl2smpl
from utils.rotate_utils import Camera_project

for squenceId in glob.glob(os.path.join(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\recordings','*')):
    os.makedirs(os.path.join(squenceId,'img'),exist_ok=True)
    for frameId in glob.glob(os.path.join(squenceId,'Color','*')):
        img = cv2.imread(frameId)
        img = cv2.flip(img,1)
        cv2.imwrite(os.path.join(squenceId,'img',os.path.basename(frameId)),img)
        

with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smpl\000_smpl_vcl.pkl', 'rb') as file:
    data = pkl.load(file)
vs,js,fs = pkl2smpl(
    R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smpl\000_smpl_vcl.pkl',
    'smpl')
vs = Camera_project(vs,data['person00']['cam_extrinsic'],data['person00']['cam_intrinsic'])
img = cv2.imread(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\images\s001_frame_00001__00.00.00.023\000.png')
for p in vs:
    img = cv2.circle(img,(int(p[0]),int(p[1])),1,(0,0,255),4)
cv2.imshow('1',img)
cv2.waitKey(0)

with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\calibration\Color.json', 'rb') as file:
    cam_color = json.load(file)

for squenceId in glob.glob(os.path.join(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
    for frameId in glob.glob(os.path.join(squenceId,'results','*')):
        with open(os.path.join(frameId, 'smpl', '000_smpl_final_cam.pkl'), 'rb') as file:
            data = pkl.load(file)
            data['person00']['cam_intrinsic'] = np.array(cam_color['camera_mtx'])
        with open(os.path.join(frameId, 'smplx', '000_final.pkl'), 'rb') as file:
            data1 = pkl.load(file)
            data1['person00']['cam_intrinsic'] = np.array(cam_color['camera_mtx'])
            data1['person00']['cam_extrinsic'] = data['person00']['cam_extrinsic']
        with open(os.path.join(frameId, 'smplx', '000_vcl.pkl'), 'wb') as file:
            pkl.dump(data1,file)



pklPath = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s002_frame_00001__00.00.00.013\smpl\000_smpl_final_cam.pkl'
with open(pklPath, 'rb') as file:
    data = pkl.load(file)
    
vs, js, fs = pkl2smpl(
    pklPath,
    'smpl'
    )

meshData = MeshData()
meshData.vert = vs
meshData.face = fs

write_obj(
    R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s002_frame_00001__00.00.00.013\test\smpl_final.obj',
    meshData
)

vs = Camera_project(vs, data['person00']['cam_extrinsic'])
meshData.vert = vs
meshData.face = fs

write_obj(
    R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s002_frame_00001__00.00.00.013\test\smpl_final_cam.obj',
    meshData
)

with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\calibration\Color.json', 'rb') as file:
    cam_color = json.load(file)
meshData.vert=Camera_project(meshData.vert, np.eye(4,4), cam_color['camera_mtx'])

img = cv2.imread(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\images\s002_frame_00001__00.00.00.013\000.png')
for p in meshData.vert:
    img = cv2.circle(img,(int(p[0]),int(p[1])),1,(0,0,255),4)
cv2.imshow('1',img)
cv2.waitKey(0)
