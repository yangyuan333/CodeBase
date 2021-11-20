import numpy as np
import json
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import pickle
import zlib
from matplotlib import pyplot as plt
import sys
import msgpack
import re
import os
import msgpack_numpy
import cv2

root_path = R'E:\Human-Training-v3.3\HumanData'

msgpack_numpy.patch()

ids = np.load(os.path.join(root_path, 'c2gimgid.npy'))
jsonfile = 'xyz_gpa12_cntind_world_cams.json'
import matplotlib.pyplot as plt

with open(os.path.join(root_path, jsonfile),'r') as f:
    annot = json.load(f)
print(annot.keys())
with open(os.path.join(root_path, 'xyz_gpa12_mdp_cntind_crop_cam_c2g.json'),'r') as f:
    annot1 = json.load(f)
print(annot.keys())
img_root_path = R'E:\Human-Training-v3.3\HumanData\Gaussian_cropped_images_greenbackground.tar\Gaussian_GPA12_img_greenbackground_cropped_version'
img_name = os.path.join(img_root_path, annot['images'][0]['file_name'][-14:-4] + '.png')

mdp_path = os.path.join(R'E:\Human-Training-v3.3\HumanData\crop_md', annot['images'][0]['file_name'][-14:-4]+'.bin')
def read_bytes(fname):
    with open(fname, "rb") as f:
        return f.read()
def load_compressed_bytes(obj, **kwargs):
    return msgpack.loads(zlib.decompress(obj), **kwargs)
def load_bytes(in_bytes, **kwargs):
    return msgpack.loads(in_bytes, **kwargs)

img_name = R'E:\Human-Training-v3.3\HumanData\gaussian_cropped_images.tar\Gaussian_cropped_images\0000000146.png'
img4 = cv2.imread(img_name)
d2_joint = np.array(annot['annotations'][0]['joint_imgs'])
#for joint in d2_joint:
#    img4 = cv2.circle(img4, (int(joint[0]), int(joint[1])), 1, (0,0,255),1,8,0)
img4 = cv2.circle(img4, (int(d2_joint[26][0]), int(d2_joint[26][1])), 1, (0,0,255),1,8,0)
img4 = cv2.circle(img4, (int(d2_joint[31][0]), int(d2_joint[31][1])), 1, (0,0,255),1,8,0)
cv2.imshow('1', img4)
cv2.waitKey(0)
print(0)