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
from utils.obj_utils import write_obj
root_path = R'E:\Human-Training-v3.3\HumanData'

msgpack_numpy.patch()

jsonfile = 'xyz_gpa12_cntind_world_cams.json'
import matplotlib.pyplot as plt

with open(os.path.join(root_path, jsonfile),'r') as f:
    annot = json.load(f)
import numpy as np
img_scence_id = np.load('E:\Human-Training-v3.3\HumanData\scene_meshes-20210910T035831Z-001\scene_meshes\img_scene_id.npy')

save_joints = []
annots = annot['annotations']
for img_annot in annots:
    joint_world = img_annot['joint_world_mm']
    save_joints.append(joint_world[26])
    save_joints.append(joint_world[31])

write_obj('GPA_DATA.obj', save_joints)