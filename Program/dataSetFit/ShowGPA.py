'''
 @FileName    : ShowGPA.py
 @EditTime    : 2021-09-10 21:55:09
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import sys
import os
import cv2
sys.path.append('./')
from utils.FileLoaders import load_json, save_json
from utils.projection import joint_projection
from utils.utils import vis_img, move
import numpy as np
from tqdm import tqdm

img2scene = np.load(r'E:\GPA\scene_meshes\img_scene_id.npy')
data = load_json(r'E:\GPA\xyz_gpa12_cntind_world_cams.json')
data_crop = load_json(r'E:\GPA\xyz_gpa12_mdp_cntind_crop_cam_c2g.json')
images = data['images']
annotations = data['annotations']
annotations_crop = data_crop['annotations']
for img, annot, sid, annot_crop in tqdm(zip(images, annotations, img2scene, annotations_crop), total=len(images)):
    # img_path = os.path.join(r'F:\GPA-fitting\images_all', img['file_name'].split('/')[-1]) 
    # img = cv2.imread(img_path)
    # joints = np.array(annot['joint_cams']).T

    extri = np.eye(4)
    intri = np.eye(3)

    # joints = np.array(annot['joint_world_mm'])
    rot = cv2.Rodrigues(np.array(annot['src_cam2']))[0]
    trans = np.array(annot['src_cam3'])

    extri[:3,:3] = rot
    extri[:3,3] = trans.reshape(3,) * 10

    # intri = np.array(annot['src_cam0'])

    intri[0][0] = annot_crop['c_f'][0]
    intri[1][1] = annot_crop['c_f'][1]
    intri[0][2] = annot_crop['c_c'][0]
    intri[1][2] = annot_crop['c_c'][1]

    src_img = os.path.join(r'F:\GPA-fitting\images_all', img['file_name'].split('/')[-1])
    dst_img = os.path.join(r'F:\GPA-fitting\images', 'TS%02d' %int(sid), img['file_name'].split('/')[-1])
    annot['crop_intri'] = intri.tolist()
    annot['extri'] = extri.tolist()
   
    dst_kep = os.path.join(r'F:\GPA-fitting\annots', 'TS%02d' %int(sid), img['file_name'].split('/')[-1].replace('png', 'json'))
    save_json(dst_kep, annot)
    move(src_img, dst_img)
    # joint2d, _ = joint_projection(joints, extri, intri, img, True)
    # print(1)

