import json
import os
import numpy as np
path = R'E:\Human-Training-v3.2\原始素材 VCL Occlusion\3DOH50K\3DOH50K-train'

with open(os.path.join(path, 'annots.json')) as f:
    datas = json.load(f)
print('load done!')

save_data = []
for data in datas.values():
    exmat = np.array(data['extri'])
    joint_3d = np.array(data['smpl_joints_3d'])
    trans = np.array(data['trans'])
    scale = np.array(data['scale'])
    joint_3d = (joint_3d-trans)/scale[0]+trans
    joint_3d_world = np.dot(np.linalg.inv(exmat), np.row_stack((joint_3d.T, np.ones(joint_3d.__len__()))))
    save_data.append([joint_3d_world[:3,7]])
    save_data.append([joint_3d_world[:3,8]])

f = open('joint.obj', 'w')
for d in save_data:
    f.write('v ' + str(d[0][0]) + ' ' + str(d[0][1]) + ' ' + str(d[0][2]) + '\n')
f.close()