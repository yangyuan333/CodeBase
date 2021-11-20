import glob
import os
import sys
import json
import numpy as np
import shutil
sys.path.append('./')
from Program.GPA import Config

imgRootPath = r'I:\BaiduNetdiskDownload\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k\img_jpg_gaussian_750k'
annotRootPath = r'E:\GPA7002'
saveRootPath = r'E:\GPA7002\GPAImg'

GPAtool = Config.GPATool(annotRootPath)

imgDatas = np.load(os.path.join(annotRootPath, 'sequence_ids', 'group_5view_new_0to5.npy'))

camEx = {
    'cam0':[],
    'cam1':[],
    'cam2':[],
    'cam3':[],
    'cam4':[],
}
camTs = {
    'cam0':[],
    'cam1':[],
    'cam2':[],
    'cam3':[],
    'cam4':[],
}
camIn = {
    'cam0':[],
    'cam1':[],
    'cam2':[],
    'cam3':[],
    'cam4':[],
}
for squenceId, squenceData in enumerate(imgDatas):
    for imgId, frameData in enumerate(squenceData):
        for camId in range(5):
            if int(frameData[camId]) == 0:
                continue
            imgName = str(int(frameData[camId])).zfill(10)
            if imgName in GPAtool.imgName2idx:
                camIn['cam'+str(camId)].append(GPAtool.annotDatas['annotations'][GPAtool.imgName2idx[imgName]]['src_cam0'])
                camEx['cam'+str(camId)].append(GPAtool.annotDatas['annotations'][GPAtool.imgName2idx[imgName]]['src_cam2'])
                camTs['cam'+str(camId)].append(GPAtool.annotDatas['annotations'][GPAtool.imgName2idx[imgName]]['src_cam3'])
import json
with open('./Program/GPA/data/In.txt', 'w') as file:
    json.dump(camIn, file)
with open('./Program/GPA/data/Ex.txt', 'w') as file:
    json.dump(camEx, file)
with open('./Program/GPA/data/Ts.txt', 'w') as file:
    json.dump(camTs, file)