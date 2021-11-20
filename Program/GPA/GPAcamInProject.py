import numpy as np
import os
import cv2
import sys
sys.path.append('./')
from utils import rotate_utils, obj_utils

path = r'D:\yy_code\Code\phy_program\CodeBase\Program\GPA\data'
camIn = np.loadtxt(os.path.join(path, 'camIn.txt'))
camExRotVec = np.loadtxt(os.path.join(path, 'camExRotVec.txt'))
camExTrans = np.loadtxt(os.path.join(path, 'camExTrans.txt'))
joint_cam = np.loadtxt(os.path.join(path, 'joint_cam.txt'))
joint_imgs = np.loadtxt(os.path.join(path, 'joint_imgs.txt'))
joint_imgs_uncrop = np.loadtxt(os.path.join(path, 'joint_imgs_uncrop.txt'))
joint_world_mm = np.loadtxt(os.path.join(path, 'joint_world_mm.txt'))
joint_world = joint_world_mm

Exr = rotate_utils.R.from_rotvec(camExRotVec)
Inr = rotate_utils.R.from_matrix([list(camIn[0]),list(camIn[1]),list(camIn[2])])

import cv2
cv2.Rodrigues(camIn)

# jointCam = Exr.apply(joint_world) + camExTrans[None,:]
# joint_cam *= 10

jointImg = (Inr.apply(joint_cam.T) / joint_cam[2,:][:,None])
meshData = obj_utils.MeshData()
jointImg[:,2] = np.ones(34)

# meshData.vert = jointImg
# obj_utils.write_obj('./Program/GPA/data/projection/world2img.obj', meshData)
# meshData.vert = np.column_stack((joint_imgs, np.ones((34,1))))
# obj_utils.write_obj('./Program/GPA/data/projection/img.obj', meshData)
# meshData.vert = np.column_stack((joint_imgs_uncrop, np.ones((34,1))))
# obj_utils.write_obj('./Program/GPA/data/projection/imguncrop.obj', meshData)

p = Inr.apply(joint_cam[:,0]) / joint_cam[2,0]
p1 = np.dot(camIn, joint_cam[:,0][:,None])/joint_cam[2,0]
print(0)
