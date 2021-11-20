import json
import os
import numpy as np
path = R'E:\Human-Training-v3.2\原始素材 VCL Occlusion\3DOH50K\3DOH50K-train'

with open(os.path.join(path, 'annots.json')) as f:
    data = json.load(f)
print('load done!')
data0 = data['00000']
exmat = np.array(data0['extri'])
inmat = np.array(data0['intri'])
joint_3d = np.array(data0['smpl_joints_3d'])
joint_2d = np.array(data0['smpl_joints_2d'])

joint_data = np.array([[joint_3d[0][0]],
                    [joint_3d[0][1]],
                    [joint_3d[0][2]],
                    [1]])
joint_3d_cam = np.dot(exmat, joint_data)
joint_2d_cam = np.dot(inmat, joint_data[:3,0])/joint_data[2,0]
print(1)