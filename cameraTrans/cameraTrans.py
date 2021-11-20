import sys
sys.path.append('./')
import pickle
from utils.obj_utils import *
from utils.rotate_utils import *

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os 

with open('./cameraTrans/H36MCameraData/joints3D.pkl', 'rb') as f:
    data = pickle.load(f)
print(1)

a = planeFit(data)

vec = np.array([0,1,0])
r = CalRotFromVecs(np.array([a[0],a[1],-1]), vec)

dataNew = r.apply(data)
# write_obj('./cameraTrans/H36MCameraData/pointsNew.obj', dataNew)

cams = [
    [
        [-0.9033486325708847, 0.4269119513835288, 0.04132110596558493, -0.3212076965010843],
        [0.04153060140040344, 0.18295116146549473, -0.982244410350981, 0.4671347939539068 ],
        [-0.4268916222879434, -0.885593054559674, -0.18299859162308746, 5.514330481231836],
        [0,0,0,1],
    ],
    [
        [0.9315720438826058, 0.3634828886598199, -0.00732916829502759, 0.019193081470653548 ],
        [0.06810071942874552, -0.19426751468589246, -0.9785818436651346, 0.40422815517048916],
        [-0.357121574634845, 0.9111203665758325, -0.2057276319337036, 5.702169449034496 ],
        [0,0,0,1],
    ],
    [
        [-0.9269344224860929, -0.37323034448550024, -0.038622355774959594, 0.45540105803335257 ],
        [-0.04725991979946154, 0.21824052930449528, -0.9747500045393394, 0.27335888905742267 ],
        [0.3722352433487741, -0.9017040430895984, -0.2199334951410318, 5.657814750065345 ],
        [0,0,0,1],
    ],
    [
        [0.9154607101090738, -0.3973460632567893, 0.06362227802367726, -0.06927120383747176],
        [-0.0494062705446705, -0.2678916037393088, -0.9621814325151199, 0.4221846819630868 ],
        [0.3993628784419721, 0.8776959579833511, -0.2648756248925708, 4.457893157362932],
        [0,0,0,1],
    ]
]

for key, cam in enumerate(cams):
    camNew = np.loadtxt(os.path.join(r'./cameraTrans/H36MCameraData', str(key)+'.txt'))
    cam = np.array(cam)
    RR = cam[:3,:3] 
    point_in_cam = R.from_matrix(RR).apply(data)
    point_New_in_cam = R.from_matrix(camNew).apply(dataNew)
    print(1)

# for key, cam in enumerate(cams):
#     cam = np.array(cam)
#     R = cam[:3,:3]
#     np.savetxt(os.path.join(r'./cameraTrans/H36MCameraData',str(key)+'.txt'), np.dot(R, np.linalg.inv(np.array(r.as_matrix()))))
