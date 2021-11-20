import os
import numpy as np
import pickle
from numpy.lib import utils
from utils.obj_utils import *
from utils.smpl_utils import SMPLModel
from utils.rotate_utils import *
import torch
import math
import glob

templt_path = 'template.obj'
templt_vs, templt_fs = read_obj(templt_path)
smpl_model = SMPLModel()

path = R'E:\Human-Training-v3.2\AMASS\annotations'
save_path = R'E:\Human-Training-v3.3\AMASS\annotations'
dirs_path = glob.glob(os.path.join(path, '*'))

def recFindFile(path, root_path, save_path):
    if os.path.isdir(path):
        dirs_path = glob.glob(os.path.join(path, '*'))
        for dir_path in dirs_path:
            recFindFile(dir_path, root_path, save_path)
    else:
        path_names = path.split('\\')
        root_names = root_path.split('\\')

        save_path_path = save_path
        for name in path_names[root_names.__len__():-1]:
            save_path_path = os.path.join(save_path_path, name)
            if not os.path.exists(save_path_path):
                os.mkdir(save_path_path)
        save_path_path = os.path.join(save_path_path, path_names[-1])
        data = np.load(path)
        poses = []
        for pose in data['poses']:
            r = R.from_rotvec(pose[:3])
            r1 = R.from_rotvec([-math.pi/2, 0, 0])
            pose[:3] = (r1*r).as_rotvec()
            poses.append(pose)
        np.save(save_path_path, poses)
recFindFile(path,path,save_path)
# for dir_path in dirs_path[3:]:
#     dir_name = os.path.basename(dir_path)
#     if not os.path.exists(os.path.join(save_path, dir_name)):
#         os.mkdir(os.path.join(save_path, dir_name))
#     dir_dirs_path = glob.glob(os.path.join(dir_path, '*'))
#     for dir_dir_path in dir_dirs_path:
#         dir_dir_name = os.path.basename(dir_dir_path)
#         if not os.path.exists(os.path.join(save_path, dir_name, dir_dir_name)):
#             os.mkdir(os.path.join(save_path, dir_name, dir_dir_name))
#         files_path = glob.glob(os.path.join(dir_dir_path, '*'))
#         for file_path in files_path:
#             file_name = os.path.basename(file_path)
#             data = np.load(file_path)
#             poses = []
#             for pose in data['poses']:
#                 r = R.from_rotvec(pose[:3])
#                 r1 = R.from_rotvec([-math.pi/2, 0, 0])
#                 pose[:3] = (r1*r).as_rotvec()
#                 poses.append(pose)
#             np.save(os.path.join(save_path, dir_name, dir_dir_name, file_name), poses)

# i = 0
# for pose in data['poses']:
#     r = R.from_rotvec(pose[:3])
#     r1 = R.from_rotvec([math.pi/2, 0, 0])
#     pose[:3] = (r1*r).as_rotvec()
#     vs, js = smpl_model(betas=torch.tensor(np.zeros((1,10))).float(), thetas=torch.tensor([pose]).float(), trans=torch.tensor(np.zeros((1,3))).float(), scale=torch.tensor([1]).float(), gR=None, lsp=False)

#     write_obj('./data/amass_test'+ str(i) +'.obj', vs.squeeze(0).numpy(), templt_fs)
#     i+=1