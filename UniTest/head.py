import glob
import os
import sys
sys.path.append('./')
import pickle as pkl
import shutil
import numpy as np
import torch
from utils.smpl_utils import SMPLModel
smplModel = SMPLModel()
from utils.obj_utils import read_obj,write_obj
meshData = read_obj(R'./data/smpl/template.obj')