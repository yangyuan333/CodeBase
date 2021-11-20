import sys
sys.path.append('./')
import numpy as np
import os
import glob
import pickle
from utils.rotate_utils import *
import json

dataPath = r'H:\YangYuan\Code\phy_program\humor-main\out\rgb_demo_use_split\Human36'
savePath = r'H:\YangYuan\Code\phy_program\CodeBase\Humor\data\resultHuman'
camPath = r'E:\Evaluations_CVPR2022\Eval_Human36M10FPS'

Human = ['S9', 'S11']
for human in Human:
    squenceIds = glob.glob(os.path.join(dataPath, human, '*'))
    for squenceId in squenceIds:
        squenceName = os.path.basename(squenceId)
        saveResultPath = os.path.join(savePath, human, squenceName)
        os.makedirs(saveResultPath, exist_ok=True)

        campath = os.path.join(camPath, human, 'camparams', squenceName, 'camparams.txt')

        camIns, camExs = readVclCamparams(campath)
        cam = np.array(camExs)[0][:3,:3]

        framesData = np.load(os.path.join(squenceId, 'final_results', 'stage3_results.npz'))
        framesLen = framesData['betas'].__len__()
        idx = 0
        for beta, tran, root_orient, pose_body in zip(framesData['betas'],framesData['trans'],framesData['root_orient'],framesData['pose_body']):
            framepkl = {}
            framepkl['betas'] = beta[:10]
            framepkl['pose_body'] = pose_body
            framepkl['root_orient'] = (R.from_matrix(np.linalg.inv(cam))*R.from_rotvec(root_orient)).as_rotvec()
            framepkl['pose'] = np.append((R.from_matrix(np.linalg.inv(cam))*R.from_rotvec(root_orient)).as_rotvec(), pose_body)
            framepkl['pose'] = np.append(framepkl['pose'], np.zeros(6))
            with open(os.path.join(saveResultPath, str(idx).zfill(5)+'.pkl'), 'wb') as f:
                pickle.dump(framepkl, f, protocol=2)
            idx += 1