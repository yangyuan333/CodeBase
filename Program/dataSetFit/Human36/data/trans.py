import sys
sys.path.append('./')
import os
import pickle
from utils import rotate_utils
from utils import obj_utils
import numpy as np
path = r'H:\YangYuan\Code\phy_program\CodeBase\Program\dataSetFit\Human36\data\Human3.6M_S11'

with open(os.path.join(path, 'joints3D_s11.pkl'), 'rb') as file:
    data = pickle.load(file)

a = rotate_utils.planeFit(data) # 法向量：(a[0], a[1], -1) 点：(0, 0, a[2])
fit_n = np.array([0,-1,0])
r = rotate_utils.CalRotFromVecs(np.array([a[0], a[1], -1]), fit_n)
pointNew = r.apply([0,0,a[2]])
t = [0, -pointNew[1], 0]

# popintNew = r.apply(data) + t
# meshData = obj_utils.MeshData()
# meshData.vert = popintNew
# obj_utils.write_obj(os.path.join(path, '111.obj'), meshData)

camIns,camExs = rotate_utils.readVclCamparams(os.path.join(path, 'camparams_S11.txt'))

camExNews = []
for camEx in camExs:
    camex = np.array(camEx)
    camex[:3,:3] = (rotate_utils.R.from_matrix(camex[:3,:3]) * r.inv()).as_matrix()
    camex[:3, 3] = camex[:3, 3] - np.array(rotate_utils.R.from_matrix(camex[:3,:3]).apply(t))
    camExNews.append(camex)
rotate_utils.writeVclCamparams(os.path.join(path, 'camparamsNew.txt'), camIns, camExNews)