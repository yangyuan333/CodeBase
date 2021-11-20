import json
import pickle
import sys
sys.path.append('./')
from utils.rotate_utils import *
from utils.smpl_utils import *
import numpy as np
import cv2
import pickle

path = r'H:\YangYuan\Code\phy_program\CodeBase\data\temdata\0000000013.pkl'
with open(path, 'rb') as file:
    data = pickle.load(file)
print(1)

with open(r'H:\YangYuan\Code\phy_program\CodeBase\data\temdata\0000000013_00.json', 'r') as file:
    jsonData = json.load(file)
print(2)