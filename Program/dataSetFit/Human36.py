import os
import numpy as np
import glob
import pickle
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel
from utils.obj_utils import write_obj, read_obj, MeshData
from utils.rotate_utils import *
import torch
import cv2

smplModel = SMPLModel()
meshdata = read_obj('./data/smpl/template.obj')

with open(R'E:\train.pkl', 'rb') as f:
    datas = pickle.load(f)

pose = datas[0][100]['0']['pose']
beta = datas[0][100]['0']['betas']
trans = datas[0][100]['0']['trans']
vs, js = smplModel(betas=torch.tensor(np.array(beta,dtype=np.float32)[None,:]), thetas=torch.tensor(np.array(pose,dtype=np.float32)[None,:]), trans=torch.tensor(np.array(trans,dtype=np.float32)[None,:]), scale=torch.tensor([1.0]))

meshdata.vert = vs.squeeze(0).numpy()
write_obj('11111111111.obj', meshdata)

cam = 'cam_0'
# save_joints = []
# for sequence in datas:
#     dirs = sequence[0]['img_path'].split('\\')
#     if dirs[-2] == cam:
#         for frame in sequence:
#             frameData = frame['0']
#             vs, js = smplModel(betas=torch.tensor(np.array(frameData['betas'],dtype=np.float32)[None,:]), thetas=torch.tensor(np.array(frameData['pose'],dtype=np.float32)[None,:]), trans=torch.tensor(np.array(frameData['trans'],dtype=np.float32)[None,:]), scale=torch.tensor([1.0]))
#             write_obj('human36test.obj', vs.squeeze(0).numpy(), temFs)
#             save_joints.append(js.squeeze(0).numpy()[10])
#             save_joints.append(js.squeeze(0).numpy()[11])
# write_obj('human36JointsTest.obj', save_joints)

root_path = R'E:\Human-Training-v3.2\Human36M_MOSH'
RRR = np.loadtxt(R'E:\Human-Training-v3.3\Human36M_MOSH/mat0.txt')
for i, sequence in enumerate(datas):
    dirs = sequence[0]['img_path'].split('\\')
    if dirs[-2] == cam:
        for j, frame in enumerate(sequence):
            img_path = os.path.join(root_path, frame['img_path'])
            frameData = frame['0']
            pose = frameData['pose']
            trans = frameData['trans'][0]

            v2, j2 = smplModel(betas=torch.tensor(frameData['betas'][None,:].astype(np.float32)), thetas=torch.tensor(np.array(frameData['pose'])[None,:].astype(np.float32)), trans=torch.tensor(np.array(frameData['trans'])[None,:].astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
            j2 = j2.squeeze(0).numpy()
            v2 = v2.squeeze(0).numpy()
            write_obj('./data/test.obj', v2, temFs)
            #v_root = j2[0] - frameData['trans'][0]

            v_2d = Camera_project(v2, RRR, np.array(frameData['intri']))
            img = cv2.imread(img_path)
            for v in v_2d:
                cv2.circle(img, (int(v[0]), int(v[1])),1,(0,0,255),3)
            cv2.imshow('1',img)
            cv2.waitKey(0)
            print(1)
            #r = R.from_rotvec(pose[:3])
            #r1 = R.from_matrix(RRR[:3,:3])
            #frameData['pose'][:3] = (R.from_matrix(RRR[:3,:3])*R.from_rotvec(frameData['pose'][:3])).as_rotvec()
            #frameData['trans'][0] = np.dot(RRR[:3,:3], np.array(trans)[:,None])[:,0] + RRR[:3,3] + np.dot(RRR[:3,:3], v_root[:,None])[:,0] - v_root
            # vs, js = smplModel(betas=torch.tensor(np.array(frameData['betas'],dtype=np.float32)[None,:]), thetas=torch.tensor(np.array(frameData['pose'],dtype=np.float32)[None,:]), trans=torch.tensor(np.array(frameData['trans']).T.astype(np.float32)), scale=torch.tensor([1.0]), gR=None, lsp=False)
            # vs = vs.squeeze(0).numpy()

            #datas[i][j]['0']['pose'] = frameData['pose']
            #datas[i][j]['0']['trans'] = frameData['trans']

            # exmat = np.eye(4)
            # exmat[:3, :3] = np.dot(exmat[:3, :3], np.linalg.inv(RRR[:3, :3]))
            # exmat[:3, 3] = exmat[:3, 3] - np.dot(exmat[:3, :3], RRR[:3, 3][:,None])[:,0]

            # vs_2d = Camera_project(v2, RRR, np.array(frameData['intri']))
            # for v in vs_2d:
            #     cv2.circle(img, (int(v[0]), int(v[1])),1,(255,0,),3)
            # cv2.imshow('1', img)
            # cv2.waitKey(0)
            # print(0)

            #r = R.from_rotvec(pose[:3])
            #r1 = R.from_matrix(RRR[:3,:3])

            #datas[i][j]['0']['pose'][:3] = (R.from_matrix(RRR[:3,:3])*R.from_rotvec(frameData['pose'][0,:3])).as_rotvec()
            #datas[i][j]['0']['trans'] = np.dot(RRR[:3,:3], trans[:,None])[:,0] + RRR[:3,3] + np.dot(RRR[:3,:3], v_root[:,None])[:,0] - v_root
            #vs, js = smplModel(betas=torch.tensor(np.array(frameData['betas'],dtype=np.float32)[None,:]), thetas=torch.tensor(np.array(datas[i][j]['0']['pose'],dtype=np.float32)[None,:]), trans=torch.tensor(datas[i][j]['0']['trans'][None,:].astype(np.float32)), scale=torch.tensor([1.0]))
            #write_obj('test_new.obj', vs.squeeze(0).numpy(), temFs)
            #datas[i][j]['0']['pose'] = pose
            #datas[i][j]['0']['trans'] = trans
# with open(R'E:\Human-Training-v3.3\Human36M_MOSH\annot/test3.pkl','wb') as f:
#     pickle.dump(datas, f)