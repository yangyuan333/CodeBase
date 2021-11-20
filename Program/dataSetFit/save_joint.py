import pickle
import cv2
import os
import numpy as np
path = R'E:\Human-Training-v3.2\VCLOcclusion'
f = open(os.path.join(path, 'annot/train.pkl'),'rb')
data = pickle.load(f)
f.close()
data_len = data.__len__()
print(data_len)
save_data = []
i = 0
for cam_data in data:
    if i > 2:
        break
    i+=1
    for people_data in cam_data:
        scale = people_data['0']['scale']
        joint_3d = np.array(people_data['0']['smpl_joints_3d'])
        trans = np.array(people_data['0']['trans'])
        save_data.append((joint_3d-trans)/scale[0]+trans)

f = open('joint.txt','w')
for jd in save_data:
    f.write('v ' + str(jd[7][0]) + ' ' + str(jd[7][1]) + ' ' + str(jd[7][2]) + '\n')
    f.write('v ' + str(jd[8][0]) + ' ' + str(jd[8][1]) + ' ' + str(jd[8][2]) + '\n')
f.close()