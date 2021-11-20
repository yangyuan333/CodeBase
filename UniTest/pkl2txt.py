import sys
sys.path.append('./')
import os
import glob
import pickle
import numpy as np

rootPath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\motionDataPKL\qulitative_results'
saveRootPath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\motionData\qulitative_results'
os.makedirs(saveRootPath, exist_ok=True)
for path in glob.glob(os.path.join(rootPath, '*')):
    pklPath = path
    savePath = saveRootPath
    with open(pklPath, 'rb') as file:
        pklData = pickle.load(file)
    if 'person00' in pklData:
        pose = pklData['person00']['pose']
        transl = pklData['person00']['transl']
    else:
        pose = pklData['pose']
        transl = pklData['transl']
    np.savetxt(os.path.join(savePath, os.path.basename(pklPath).split('.')[0]+'_pose.txt'), pose)
    np.savetxt(os.path.join(savePath, os.path.basename(pklPath).split('.')[0]+'_transl.txt'), transl)