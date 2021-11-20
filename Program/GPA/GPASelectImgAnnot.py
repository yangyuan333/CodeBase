import glob
import os
import sys
import json
import numpy as np
import shutil
import pickle
sys.path.append('./')
from Program.GPA import Config

imgRootPath = r'I:\BaiduNetdiskDownload\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k\img_jpg_gaussian_750k'
annotRootPath = r'E:\GPA7002'
saveRootPath = r'E:\GPA7002\GPAImg'
saveAnootPath = r'E:\GPA7002\GPAAnnot'
GPAtool = Config.GPATool(annotRootPath)

imgDatas = np.load(os.path.join(annotRootPath, 'sequence_ids', 'group_5view_new_0to5.npy'))

saveDatas = []
for squenceId, squenceData in enumerate(imgDatas[24:]):
    squenceId += 24
    squenceSaveData = []
    for imgId, frameData in enumerate(squenceData):
        for camId in range(5):
            if int(frameData[camId]) == 0:
                continue
            frameSaveData = {}
            frameSaveData['camId'] = camId
            imgName = str(int(frameData[camId])).zfill(10)
            imgPath = os.path.join(imgRootPath, imgName+'.jpg')
            os.makedirs(os.path.join(saveRootPath, 'all', str(squenceId).zfill(4), 'Camera'+str(camId).zfill(2)), exist_ok=True)
            if os.path.exists(imgPath):
                shutil.copyfile(imgPath, os.path.join(saveRootPath, 'all', str(squenceId).zfill(4), 'Camera'+str(camId).zfill(2), str(imgId).zfill(10)+'.jpg'))
                frameSaveData['img_path'] = os.path.join(saveRootPath, 'all', str(squenceId).zfill(4), 'Camera'+str(camId).zfill(2), str(imgId).zfill(10)+'.jpg')
            else:
                print(imgPath)
                frameSaveData['img_path'] = ''
            frameSaveData['0'] = {}
            if imgName in GPAtool.imgName2idx:
                frameSaveData['0']['joint_world'] = GPAtool.annotDatas['annotations'][GPAtool.imgName2idx[imgName]]['joint_world_mm']
                frameSaveData['0']['joint_img'] = GPAtool.annotDatas['annotations'][GPAtool.imgName2idx[imgName]]['joint_imgs_uncrop']
            squenceSaveData.append(frameSaveData)
    saveDatas.append(squenceSaveData)

with open(os.path.join(saveAnootPath, 'all.pkl'), 'wb') as file:
    pickle.dump(saveDatas, file, protocol=2)