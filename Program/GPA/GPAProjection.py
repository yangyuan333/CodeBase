import sys
sys.path.append('./')
import cv2
import os
import glob
from Program.GPA import Config
from utils import obj_utils
from utils import rotate_utils
import numpy as np
imgRootPath = r'I:\BaiduNetdiskDownload\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k.tar.gz\img_jpg_gaussian_750k\img_jpg_gaussian_750k'
RootPath = r'E:\GPA7002'
GPAtool = Config.GPATool(RootPath)

for imgData, annotData in zip(GPAtool.annotDatas['images'], GPAtool.annotDatas['annotations']):
    imgName = os.path.basename(imgData['file_name'])[:-4]
    imgPath = os.path.join(imgRootPath, imgName+'.jpg')

    joint_cam = annotData['joint_cams']
    joint_world_mm = annotData['joint_world_mm']
    joint_imgs = annotData['joint_imgs']
    joint_imgs_uncrop = annotData['joint_imgs_uncrop']
    camIn = annotData['src_cam0']
    camExRotVec = annotData['src_cam2']
    camExTrans = annotData['src_cam3']
    print(1)

    # np.savetxt('./Program/GPA/data/joint_world_mm.txt', np.array(joint_world_mm))
    # np.savetxt('./Program/GPA/data/camExRotVec.txt', np.array(camExRotVec))
    # np.savetxt('./Program/GPA/data/camExTrans.txt', np.array(camExTrans))
    # np.savetxt('./Program/GPA/data/joint_cam.txt', np.array(joint_cam))
    # np.savetxt('./Program/GPA/data/joint_imgs.txt', np.array(joint_imgs))
    # np.savetxt('./Program/GPA/data/joint_imgs_uncrop.txt', np.array(joint_imgs_uncrop))
    # np.savetxt('./Program/GPA/data/camIn', np.array(camIn))

    # 外参矩阵测试
    # r = rotate_utils.R.from_rotvec([camExRotVec[0][0],camExRotVec[1][0],camExRotVec[2][0]])
    # points_cam = np.array(r.apply(joint_world_mm)) + 10 * np.array(camExTrans).T
    # meshData = obj_utils.MeshData()
    # meshData.vert = points_cam
    # obj_utils.write_obj('./Program/GPA/data/jointworld2cam.obj', meshData)
    # meshData.vert = list(np.array(joint_cam).T)
    # obj_utils.write_obj('./Program/GPA/data/jointcam.obj', meshData)

    # r = rotate_utils.R.from_matrix([camIn]) ## 库存在bug，转换出错
    # points_img = r.apply(np.array(joint_cam).T) / np.array(joint_cam)[2,:][:,None]
    points_img = np.dot(np.array(camIn), np.array(joint_cam)) / np.array(joint_cam)[2,:][None,:]

    points_img = points_img.T
    img = cv2.imread(imgPath)
    for joint in points_img:
        img = cv2.circle(img, (int(joint[0]),int(joint[1])), 2, (255,0,0))
    cv2.imshow('1', img)
    cv2.waitKey(0)