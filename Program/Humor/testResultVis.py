import torch
import cv2
import sys
sys.path.append('./')
import numpy as np
import os
import glob
from Program.Humor.body_model import body_model
from utils.obj_utils import write_obj, read_obj
path = r'./Humor/data'
HumorPath = os.path.join(path, 'output')

VCL = ['0013', '0027', '0029']

resultPath = r'H:\YangYuan\Code\phy_program\humor-main\out\rgb_demo_use_split\GPA\0052\final_results'
imgPath = r'E:\Evaluations_CVPR2022\Eval_GPA\images\0052\Camera00'
data = np.load(os.path.join(resultPath, 'stage3_results.npz'))
gt = np.load(os.path.join(resultPath, 'gt_results.npz'))

smplH = body_model.BodyModel(
    bm_path=r'./Program/Humor/body_model/model/male.npz',
    num_betas=16,
    batch_size=1,
    use_vtx_selector=True
)
# root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, 
# betas=None, trans=None, dmpls=None, expression=None, return_dict=Falseprint(1)

# out = smplH(
#     root_orient=torch.tensor(data['root_orient'][0][None,:]), 
#     pose_body=torch.tensor(data['pose_body'][0][None,:]), 
#     betas=torch.tensor(data['betas'][0][None,:]),
#     trans=torch.tensor(data['trans'][0][None,:]), 
#     return_dict=True, 
#     )
print(1)

meshData = read_obj('./data/smpl/template.obj')

# img = cv2.imread(r'E:\Evaluations_CVPR2022\Eval_GPA\images\0052\Camera00\0000000000.jpg')

# cam = np.array(
#     [
#         [1071.506958565541, 0.0, 970.0303499730985],
#         [0.0, 1077.113483765303, 537.2649362251441],
#         [0.0, 0.0, 1.0],
#     ]
#     )

cam = np.array(gt['cam_mtx'])

for idx, imgpath in enumerate(glob.glob(os.path.join(imgPath, '*'))):
    img = cv2.imread(imgpath)

    out = smplH(
        root_orient=torch.tensor(data['root_orient'][idx][None,:]), 
        pose_body=torch.tensor(data['pose_body'][idx][None,:]), 
        betas=torch.tensor(data['betas'][idx][None,:]),
        trans=torch.tensor(data['trans'][idx][None,:]), 
        return_dict=True, 
        )

    for v in out['v'].detach().numpy()[0]:
        vc = np.dot(cam, v[:,None])/v[2]
        img = cv2.circle(img, (int(vc[0][0]), int(vc[1][0])), 2, (255,0,0))
    cv2.imshow('1', img)
    cv2.waitKey(0)
# write_obj('humor.obj', out['v'].detach().numpy()[0], f)