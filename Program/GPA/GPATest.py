import glob
import os
import sys
import json
import numpy as np
sys.path.append('./')

imgRootPath = r'I:\BaiduNetdiskDownload\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k\img_jpg_gaussian_750k'
annotRootPath = r'E:\GPA7002'

with open(os.path.join(annotRootPath, 'xyz_gpa12_cntind_world_cams.json'), 'r') as file:
    annotDatas = json.load(file)
with open(os.path.join(annotRootPath, 'xyz_gpa12_mdp_cntind_crop_cam_c2g.json'), 'r') as file:
    annotDatas1 = json.load(file)
imgData = np.load(os.path.join(annotRootPath, 'sequence_ids', 'group_5view_new_0to5.npy'))
imgData1 = np.load(os.path.join(annotRootPath, 'sequence_ids', 'cam_ids_750k.npy'))