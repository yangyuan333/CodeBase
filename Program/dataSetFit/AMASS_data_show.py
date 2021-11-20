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

path = R'E:\Human-Training-v3.3\AMASS\annotations\BMLhandball\S01_Expert'
files_path = glob.glob(os.path.join(path, '*'))
i = 0
for file_path in files_path:
    data = np.load(file_path)
    #pose = data['poses'][0]
    pose = data[0]
    #r = R.from_rotvec(pose[:3])
    #r1 = R.from_rotvec([-math.pi/2, 0, 0])
    #pose[:3] = (r1*r).as_rotvec()
    vs, js = smpl_model(betas=torch.tensor(np.zeros((1,10))).float(), thetas=torch.tensor([pose]).float(), trans=torch.tensor(np.zeros((1,3))).float(), scale=torch.tensor([1]).float(), gR=None, lsp=False)
    write_obj('./data/amass_test'+ str(i) +'.obj', vs.squeeze(0).numpy(), templt_fs)
    i+=1
# i = 0
# for pose in data['poses']:
#     r = R.from_rotvec(pose[:3])
#     r1 = R.from_rotvec([math.pi/2, 0, 0])
#     pose[:3] = (r1*r).as_rotvec()
#     vs, js = smpl_model(betas=torch.tensor(np.zeros((1,10))).float(), thetas=torch.tensor([pose]).float(), trans=torch.tensor(np.zeros((1,3))).float(), scale=torch.tensor([1]).float(), gR=None, lsp=False)

#     write_obj('./data/amass_test'+ str(i) +'.obj', vs.squeeze(0).numpy(), templt_fs)
#     i+=1