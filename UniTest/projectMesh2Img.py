import cv2
import sys
import os
sys.path.append('./')
from utils import obj_utils, rotate_utils
import numpy as np
path = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\supmat_results_all'
img = cv2.imread(os.path.join(path, '3DOH0013_Camera04_00076.jpg'))

meshData = obj_utils.read_obj(os.path.join(path, '3DOH0013_Camera04_00076_ours.obj'))
vs = meshData.vert

camIns, camExs = rotate_utils.readVclCamparams(os.path.join(path, '3DOH0013_Camera04_00076.txt'))
camIn, camEx = camIns[4], camExs[4]

for v in vs:
    v_img = rotate_utils.Camera_project(np.array([v]), np.array(camEx), np.array(camIn))
    img = cv2.circle(img, (int(v_img[0][0]), int(v_img[0][1])), 2,(255,0,0))
cv2.imshow('1', img)
cv2.waitKey(0)