import numpy as np
import glob
import os
import shutil

path = 'E:\Experiments_CVPR2022'

dataPaths = glob.glob(os.path.join(path, 'data', '*'))
for dataPath in dataPaths:
    fileName = os.path.basename(dataPath)
    imgsPath = glob.glob(os.path.join(dataPath,'*'))
    savePath = os.path.join(path, 'img_data', fileName)
    os.makedirs(savePath, exist_ok=True)
    for imgPath in imgsPath:
        if imgPath.endswith('.jpg'):
            shutil.copy(imgPath, savePath)