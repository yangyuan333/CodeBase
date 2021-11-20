import numpy as np
import sys
sys.path.append('./')
import os
from utils import obj_utils

path = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf\0000000130.obj'
# color1 = np.array([int('28',16), int('30',16), int('48',16)]) # down
# color2 = np.array([int('85',16), int('93',16), int('98',16)]) # top
# color1 = np.array([int('3d',16), int('30',16), int('48',16)]) # down
# color2 = np.array([int('85',16), int('93',16), int('98',16)]) # top
color1 = np.array([int('1f',16), int('1c',16), int('2c',16)]) # down
color2 = np.array([int('92',16), int('8d',16), int('ab',16)]) # top
meshData = obj_utils.read_obj(path)
vs, fs = np.array(meshData.vert), np.array(meshData.face)

vs_min, vs_max = vs.min(0), vs.max(0)
print(vs_min, vs_max)

colors = []

for v in vs:
    a = 1 - (v[1]-vs_min[1])/(vs_max[1]-vs_min[1])
    a = a
    b = 1 - a
    colors.append((color1*a + color2*b)/255)

meshData.color = colors
obj_utils.write_obj(r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf/color.obj', meshData)