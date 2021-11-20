import pickle
path = 'H:\YangYuan\Code\phy_program\program_1\SMPL_python_v.1\SMPL_python_v.1.1.0\smpl\models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl'
f = open(path,'rb')
data = pickle.load(f, encoding='latin')

img_data = data[0][0]
print(img_data)