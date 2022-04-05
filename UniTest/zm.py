import sys
sys.path.append('./')
import numpy as np
import pickle as pkl
import json
from scipy.spatial.transform import Rotation as Rot
from utils.obj_utils import MeshData,write_obj
from utils.rotate_utils import Camera_project, GetRotFromVecs
from utils.smpl_utils import pkl2smpl
import os
import glob

def proposeSmpl(verts,joints):
    ## joint[0] ~ joint[12] 作为竖直方向 [0,1,0]
    ## joint[17] ~ joint[16] 作为水平方向 [1,0,0]
    ## 归一化 中心移原点，大小缩放1
    ## verts: N*3
    ## joints: N*3

    yaixs = np.array([0,1,0]).astype(np.float32)
    xaixs = np.array([1,0,0]).astype(np.float32)

    ## 竖直旋转
    root = joints[0].copy()
    model_aixs1 = joints[12] - joints[0]
    rMatrix = GetRotFromVecs(
        model_aixs1, yaixs
    )
    verts = Camera_project(
        verts-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    verts += root
    joints = Camera_project(
        joints-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    joints += root

    ## 水平旋转
    root = joints[0].copy()
    model_aixs2 = joints[16] - joints[17]
    model_aixs2[1] = 0.0
    rMatrix = GetRotFromVecs(
        model_aixs2, xaixs
    )
    verts = Camera_project(
        verts-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    verts += root
    joints = Camera_project(
        joints-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    joints += root

    ## 归一化
    bias = 0.5 * (verts.max(axis = 0) + verts.min(axis = 0))
    verts -= bias
    joints -= bias
    scale = 1 / np.max(np.linalg.norm(verts, axis = 1))
    verts = scale * verts
    joints = scale * joints

    return verts, joints

def proposeSmplx(verts,joints):
    ## joint[0] ~ joint[12] 作为竖直方向 [0,1,0]
    ## joint[17] ~ joint[16] 作为水平方向 [1,0,0]
    ## 归一化 中心移原点，大小缩放1
    ## verts: N*3
    ## joints: N*3

    yaixs = np.array([0,1,0]).astype(np.float32)
    xaixs = np.array([1,0,0]).astype(np.float32)

    ## 竖直旋转
    root = joints[0].copy()
    model_aixs1 = joints[12] - joints[0]
    rMatrix = GetRotFromVecs(
        model_aixs1, yaixs
    )
    verts = Camera_project(
        verts-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    verts += root
    joints = Camera_project(
        joints-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    joints += root

    ## 水平旋转
    root = joints[0].copy()
    model_aixs2 = joints[16] - joints[17]
    model_aixs2[1] = 0.0
    rMatrix = GetRotFromVecs(
        model_aixs2, xaixs
    )
    verts = Camera_project(
        verts-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    verts += root
    joints = Camera_project(
        joints-root,
        np.vstack((np.hstack((rMatrix,np.zeros((3,1)))),np.array([[0,0,0,1]])))
    )
    joints += root

    ## 归一化
    bias = 0.5 * (verts.max(axis = 0) + verts.min(axis = 0))
    verts -= bias
    joints -= bias
    scale = 1 / np.max(np.linalg.norm(verts, axis = 1))
    verts = scale * verts
    joints = scale * joints

    return verts, joints

if __name__ == '__main__':
    # with open(R'\\105.1.1.3\Hand\SkelEmbedding\BodyScan\exported_mesh2skel_smplx_yangyuang\vicon_03301_01_s002.pkl','rb') as file:
    #     data = pkl.load(file)
    
    # meshData = MeshData()
    # meshData.vert = data['joints']
    # write_obj('test1.obj',meshData)

    rootPath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\datasets\prox_quantitative\ground_truth'
    kinds = ['smpl','smplx']
    for kind in kinds:
        if kind == 'smpl':
            savePath = R'\\105.1.1.3\Hand\SkelEmbedding\BodyScan\exported_mesh2skel_smpl24_yangyuang'
        else:
            savePath = R'\\105.1.1.3\Hand\SkelEmbedding\BodyScan\exported_mesh2skel_smplx_yangyuang'
        for seq in glob.glob(os.path.join(rootPath,kind,'*')):
            seqName = os.path.basename(seq)
            for frame in glob.glob(os.path.join(seq,'*')):
                frameName = os.path.basename(frame)[:4]
                pklPath = glob.glob(os.path.join(frame,'*'))[0]
                vs,js,fs = pkl2smpl(pklPath,kind)
                vs,js = proposeSmpl(vs,js)
                if kind == 'smpl':
                    data = {
                        'verts':vs.astype(np.float32),
                        'faces':fs.astype(np.int32)-1,
                        'vcolors':None,
                        'joints':js.astype(np.float32)[:24]
                    }
                else:
                    data = {
                        'verts':vs.astype(np.float32),
                        'faces':fs.astype(np.int32)-1,
                        'vcolors':None,
                        'joints':js.astype(np.float32)
                    }
                with open(os.path.join(savePath,seqName+'_'+frameName+'.pkl'),'wb') as file:
                    pkl.dump(data,file)

    vs,js,fs = pkl2smpl(
        R'\\105.1.1.2\Body\NIPS2022_PhysContact\datasets\prox_quantitative\ground_truth\smplx\vicon_03301_01\s006_frame_00001__00.00.00.009\000_smplx_vcl.pkl',
        'smplx',
        gender='male')
    vs,js = proposeSmpl(vs,js)
    meshData = MeshData()
    meshData.vert = vs
    write_obj('test_verts.obj',meshData)
    meshData = MeshData()
    meshData.vert = js
    write_obj('test_joints.obj',meshData)

    ## check