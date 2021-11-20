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

# with open(os.path.join(annotRootPath, 'xyz_gpa12_mdp_cntind_crop_cam_c2g.json'), 'r') as file:
#     annotDatas = json.load(file)

imgDatas = np.load(os.path.join(annotRootPath, 'sequence_ids', 'group_5view_new_0to5.npy'))

for squenceId, squenceData in enumerate(imgDatas):
    for imgId, frameData in enumerate(squenceData):
        for camId in range(5):
            if int(frameData[camId]) == 0:
                continue
            imgName = str(int(frameData[camId])).zfill(10)
            imgPath = os.path.join(imgRootPath, imgName+'.jpg')
            if imgName in GPAtool.imgName2idx:
                if GPAtool.annotDatas['annotations'][GPAtool.imgName2idx[imgName]]['istrains']:
                    os.makedirs(os.path.join(saveRootPath, 'train', str(squenceId).zfill(4), 'camera'+str(camId).zfill(2)), exist_ok=True)
                    shutil.copyfile(imgPath, os.path.join(saveRootPath, 'train', str(squenceId).zfill(4), 'camera'+str(camId).zfill(2), str(imgId).zfill(10)+'.jpg'))
                elif GPAtool.annotDatas['annotations'][GPAtool.imgName2idx[imgName]]['istests']:
                    os.makedirs(os.path.join(saveRootPath, 'test', str(squenceId).zfill(4), 'camera'+str(camId).zfill(2)), exist_ok=True)
                    shutil.copyfile(imgPath, os.path.join(saveRootPath, 'test', str(squenceId).zfill(4), 'camera'+str(camId).zfill(2), str(imgId).zfill(10)+'.jpg'))
            # if annotDatas['images'][int(frameData[camId])-1]['image_id'] == int(frameData[camId]):
            #     imgName = os.path.basename(annotDatas['images'][int(frameData[camId])-1]['file_name'])[:-4]+'.jpg'
            #     imgPath = os.path.join(imgRootPath, imgName)
            #     if annotDatas['annotations'][int(frameData[camId])-1]['istrains']:
            #         os.makedirs(os.path.join(saveRootPath, 'train', str(squenceId), 'camera'+str(camId)), exist_ok=True)
            #         shutil.copyfile(imgPath, os.path.join(saveRootPath, 'train', str(squenceId), 'camera'+str(camId), str(imgId).zfill(10)+'.jpg'))
            #     elif annotDatas['annotations'][int(frameData[camId])-1]['istests']:
            #         os.makedirs(os.path.join(saveRootPath, 'test', str(squenceId), 'camera'+str(camId)), exist_ok=True)
            #         shutil.copyfile(imgPath, os.path.join(saveRootPath, 'test', str(squenceId), 'camera'+str(camId), str(imgId).zfill(10)+'.jpg'))
            # else:
            #     print(False)