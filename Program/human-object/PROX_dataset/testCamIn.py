import numpy as np
import cv2
import json
import sys
sys.path.append('./')
from utils.obj_utils import MeshData, read_obj, write_obj
from utils.rotate_utils import Camera_project
from utils.smpl_utils import pkl2smpl

with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\vicon2scene.json', 'rb') as file:
    cam_vicon2scene = json.load(file)
with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\cam2world\vicon.json', 'rb') as file:
    cam_cam2world = json.load(file)
with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\calibration\Color.json', 'rb') as file:
    cam_color = json.load(file)

vs, js, fs = pkl2smpl(
    R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000n.pkl',
    'SMPLX'
)
meshData = MeshData()
meshData.vert = vs

meshData.vert=Camera_project(meshData.vert, cam_vicon2scene)
meshData.vert=Camera_project(meshData.vert, np.linalg.inv(np.array(cam_cam2world)))
meshData.vert=Camera_project(meshData.vert, np.eye(4,4), cam_color['camera_mtx'])

img = cv2.imread(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\recordings\vicon_03301_01\Color\s001_frame_00001__00.00.00.023_flip.jpg')
for p in meshData.vert:
    img = cv2.circle(img,(int(p[0]),int(p[1])),1,(0,0,255),4)
cv2.imshow('1',img)
cv2.waitKey(0)