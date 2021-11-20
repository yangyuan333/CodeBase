import sys
sys.path.append('./')
import numpy as np
import glob
import os

from utils import obj_utils
from utils import rotate_utils

scence, _ = obj_utils.read_obj(r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTA-test\GTAtest.obj')
scenceNew, _ = obj_utils.read_obj(r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTA-test\new\newScence.obj')
scence = np.array(scence)
scenceNew = np.array(scenceNew)

camins, camexs = rotate_utils.readVclCamparams(r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTA-test\camparams\TS1\Camera00\02284.txt')
caminsNew, camexsNew = rotate_utils.readVclCamparams(r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTA-test\new\camparams\TS1\Camera00\02284.txt')

scenceCam = np.dot(np.array(camexs[0])[:3,:3], scence.T) + np.array(camexs[0])[:3,3][:,None]
scenceNewCam = np.dot(np.array(camexsNew[0])[:3,:3], scenceNew.T) + np.array(camexsNew[0])[:3,3][:,None]

obj_utils.write_obj('./GTA/scenceCam.obj', scenceCam.T)
obj_utils.write_obj('./GTA/scenceNewCam.obj', scenceNewCam.T)