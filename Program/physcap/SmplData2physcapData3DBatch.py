import pickle
import os
import glob
import sys
sys.path.append('./')

from utils.rotate_utils import *
import Program.physcap.smplInPhyscap as smplInPhyscap
# path = r'.\physcap\huang_elal_3DOH\results\0027'
# savePath = r'.\physcap\huang_elal_3DOH\inputData\0027.txt'

rootPath = r'./data/Physcap/Human36/3Djoints'
savePath = r'./data/Physcap/Human36/inputData'
kindPaths = [os.path.join(rootPath, 'S9'), os.path.join(rootPath, 'S11')]
for kindPath in kindPaths:
    squenceIds = glob.glob(os.path.join(kindPath, '*'))
    for squenceId in squenceIds:
        os.makedirs(os.path.join(savePath, os.path.basename(kindPath), os.path.basename(squenceId)), exist_ok=True)
        motionDatas = []
        for dataPkl in glob.glob(os.path.join(squenceId,'*')):
            print(dataPkl)
            with open(dataPkl, 'rb') as f:
                data = pickle.load(f)
            poses = data['person00']['pose'] # 72维
            trans = data['person00']['transl'] # 3维
            eulas = []
            for i in range(24):
                eulas.append(R.from_rotvec(poses[(i*3):(i*3+3)]).as_euler('XYZ', degrees=False))
            motionData = []
            motionData += list(trans.data)
            for value, dofs in zip(smplInPhyscap.PhyscapSkeletonInSmpl.values(), smplInPhyscap.PhyscapSkeletonInSmplDim.values()):
                for k, dof in enumerate(dofs):
                    if dof:
                        motionData.append(eulas[value][k])
            motionDatas.append(motionData)
        file = open(os.path.join(savePath, os.path.basename(kindPath), os.path.basename(squenceId), str(0).zfill(10)+'.txt'), 'w')
        file.write('Skeletool Motion File V1.0\n')
        for frame, motionData in enumerate(motionDatas):
            file.write(str(frame)+' ')
            for poseData in motionData:
                file.write(str(poseData)+' ')
            file.write('\n')