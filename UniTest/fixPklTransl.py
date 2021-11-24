import sys
sys.path.append('./')
import os
import glob
import pickle
import numpy as np
from utils.rotate_utils import *
import shutil

# ourPath = r'\\105.1.1.1\Body\CVPR2022\0029_full_frc3_SMPL'
# gtPath = r'\\105.1.1.112\Results_CVPR2022\kinematic-multiview-3DOH\results\0029'
# offsetsum = np.array([0,0,0]).astype(np.float64)
# for ourpkl in glob.glob(os.path.join(ourPath, '*')):
#     with open(ourpkl, 'rb') as file:
#         ourdata = pickle.load(file)
#     with open(os.path.join(gtPath, os.path.basename(ourpkl)), 'rb') as file:
#         gtdata = pickle.load(file)
#     offsetsum += (gtdata['person00']['transl'] - ourdata['person00']['transl'])
# offsetmean = offsetsum / 1199
# print(offsetmean)

offsetmean = np.array([-0.00214386,0.16099363,0.03338691])
ourPath = r'\\105.1.1.1\Body\CVPR2022\0029_full_frc3_SMPL'
savePath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\renderImg\pkl\0029_full_frc3_SMPL'
os.makedirs(savePath, exist_ok=True)
for ourpkl in glob.glob(os.path.join(ourPath, '*')):
    with open(ourpkl, 'rb') as file:
        ourdata = pickle.load(file)
    ourdata['person00']['transl'] += offsetmean
    with open(os.path.join(savePath, os.path.basename(ourpkl)), 'wb') as file:
        pickle.dump(ourdata, file)