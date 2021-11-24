import sys
sys.path.append('./')
import os
import glob
import pickle
import numpy as np
from utils.rotate_utils import *
import shutil

# ourPath = r'\\105.1.1.1\Body\CVPR2022\0027_full_frc3_SMPL'
# gtPath = r'\\105.1.1.112\Results_CVPR2022\kinematic-multiview-3DOH\results\0027'
# offsetsum = np.array([0,0,0]).astype(np.float64)
# gtPahts = glob.glob(os.path.join(gtPath,'*'))
# idx = 0
# for ourpkl in glob.glob(os.path.join(ourPath, '*')):
#     with open(ourpkl, 'rb') as file:
#         ourdata = pickle.load(file)
#     with open(gtPahts[idx], 'rb') as file:
#         gtdata = pickle.load(file)
#     idx += 1
#     offsetsum += (gtdata['person00']['transl'] - ourdata['person00']['transl'])
# offsetmean = offsetsum / idx
# print(offsetmean)

offsetmean = np.array([-0.00220671, 0.15058126, 0.0333854])
ourPath = r'\\105.1.1.1\Body\CVPR2022\0027_full_frc3_SMPL'
savePath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\pkl\0027_full_frc3_SMPL'
if os.path.exists(savePath):
    shutil.rmtree(savePath)
os.makedirs(savePath, exist_ok=True)
for ourpkl in glob.glob(os.path.join(ourPath, '*')):
    with open(ourpkl, 'rb') as file:
        ourdata = pickle.load(file)
    ourdata['person00']['transl'] += offsetmean
    with open(os.path.join(savePath, os.path.basename(ourpkl)), 'wb') as file:
        pickle.dump(ourdata, file)