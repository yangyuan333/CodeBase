import os
import numpy as np
import glob
import pickle
from utils.smpl_utils import SMPLModel
from utils.obj_utils import write_obj, read_obj
from utils.rotate_utils import *
import torch
import cv2

RRR = np.loadtxt('Human36RT3.txt')

exmat = np.eye(4)
exmat[:3, :3] = np.dot(exmat[:3, :3], np.linalg.inv(RRR[:3, :3]))
exmat[:3, 3] = exmat[:3, 3] - np.dot(exmat[:3, :3], RRR[:3, 3][:,None])[:,0]

np.savetxt(R'E:\Human-Training-v3.3\Human36M_MOSH/mat3.txt', exmat)