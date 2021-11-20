import sys
sys.path.append('./')
import numpy as np
import os
import glob
import pickle
from utils.rotate_utils import *
path = './Humor/data'
dataPath = os.path.join(path, 'output')
savePath = os.path.join(path, 'result')

cams = np.array([
    [
        [1.647480490561696320e-01, -5.920818220871540416e-02, 9.845569897604451448e-01],
        [-2.811336085316997746e-01, -9.596093715587274975e-01, -1.066521388759986144e-02],
        [9.454215868018357449e-01, -2.750349869118076662e-01, -1.747391877108037395e-01],
    ],
    [
        [1.647480490561696320e-01, -5.920818220871540416e-02, 9.845569897604451448e-01],
        [-2.811336085316997746e-01, -9.596093715587274975e-01, -1.066521388759986144e-02],
        [9.454215868018357449e-01, -2.750349869118076662e-01, -1.747391877108037395e-01],
    ],
    [
        [1.647480490561696320e-01, -5.920818220871540416e-02, 9.845569897604451448e-01],
        [-2.811336085316997746e-01, -9.596093715587274975e-01, -1.066521388759986144e-02],
        [9.454215868018357449e-01, -2.750349869118076662e-01, -1.747391877108037395e-01],
    ],
])

squenceIds = glob.glob(os.path.join(dataPath, '*'))
for squenceId, cam in zip(squenceIds, cams):
    squenceName = os.path.basename(squenceId)
    saveResultPath = os.path.join(savePath, squenceName)
    os.makedirs(saveResultPath, exist_ok=True)

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