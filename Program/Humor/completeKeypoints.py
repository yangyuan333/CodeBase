import sys
import os
import glob
import json
import numpy as np
sys.path.append('./')

path = r'E:\Evaluations_CVPR2022\Eval_GPA'

with open(os.path.join(path, 'keypoints', '0034', 'Camera00', '0000000000_keypoints.json'), 'rb') as file:
    data = json.load(file)
print(1)

squenceIds = glob.glob(os.path.join(path, 'images', '*'))

for squenceId in squenceIds:
    camIds = glob.glob(os.path.join(squenceId, '*'))
    for camId in camIds:
        imgPaths = glob.glob(os.path.join(camId, '*'))
        keypointPaths = glob.glob(os.path.join(path, 'keypoints', os.path.basename(squenceId), os.path.basename(camId), '*'))
        if imgPaths.__len__() == keypointPaths.__len__():
            continue
        keypointlast = -1
        keypoint = 0
        for imgpath in imgPaths:
            imgname = os.path.splitext(os.path.basename(imgpath))[0]
            keyname = os.path.splitext(os.path.basename(keypointPaths[keypoint]))[0]
            if (imgname+'_keypoints') == keyname:
                keypointlast = keypoint
                keypoint += 1
                continue
            else:
                with open(keypointPaths[keypointlast], 'rb') as file:
                    lastdata = json.load(file)
                with open(keypointPaths[keypoint], 'rb') as file:
                    nowdata = json.load(file)
                nowdata['people'][0]['pose_keypoints_2d'] = list(0.5*np.array(nowdata['people'][0]['pose_keypoints_2d']) + 0.5*np.array(lastdata['people'][0]['pose_keypoints_2d']))
                with open(os.path.join(path, 'keypoints', os.path.basename(squenceId), os.path.basename(camId), imgname+'_keypoints'+'.json'), 'w') as file:
                    file.write(json.dumps(nowdata))
                    # json.dump(file, nowdata)