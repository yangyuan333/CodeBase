import glob
import os
from black import out
import numpy as np
import json
import sys
sys.path.append('./')
from utils.rotate_utils import fitPlane, rotatePlane, transPlane, Camera_project
from utils.smpl_utils import applyRot2Smpl, pkl2smpl
from utils.obj_utils import MeshData, read_obj, write_obj

def Unitest_FitPlaneSmpl():
    a = fitPlane(
        vertsPath=R'vs.txt',
        flag='y',
        savePath=R'plane.obj'
    )
    pklPaths = []
    savePaths = []
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            pklPaths.append(os.path.join(frameDir,'000_scene.pkl'))
            savePaths.append(os.path.join(frameDir, 'smplx', '000_rot.pkl'))
    Rotm = rotatePlane(
        a,
        pklPaths=pklPaths,
        temRotPaths=savePaths,
        mode='smplx',
        flag='y',
    )
    pklPaths = []
    savePaths = []
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            pklPaths.append(os.path.join(frameDir, 'smplx', '000_rot.pkl'))
            savePaths.append(os.path.join(frameDir, 'smplx', '000_final.pkl'))
    Tm = transPlane(
        temRotPaths=pklPaths,
        savePaths=savePaths,
        mode='smplx',
    )
    print(Rotm)
    print(Tm)

def Unitest_FitPlaneSmpl_Frame():
    a = fitPlane(
        vertsPath=R'vs.txt',
        flag='y',
        savePath=R'plane.obj'
    )
    pklPaths = []
    savePaths = []
    # for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
    #     for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
    #         pklPaths.append(os.path.join(frameDir,'000_scene.pkl'))
    #         savePaths.append(os.path.join(frameDir, 'smplx', '000_rot.pkl'))
    pklPaths.append(R'H:\YangYuan\ProjectData\contact_duiqi.pkl')
    savePaths.append(R'H:\YangYuan\ProjectData\contact_rot.pkl')
    Rotm = rotatePlane(
        a,
        pklPaths=pklPaths,
        temRotPaths=savePaths,
        mode='smplx',
        flag='y',
    )
    pklPaths = []
    savePaths = []
    # for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
    #     for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
    #         pklPaths.append(os.path.join(frameDir, 'smplx', '000_rot.pkl'))
    #         savePaths.append(os.path.join(frameDir, 'smplx', '000_final.pkl'))
    pklPaths.append(R'H:\YangYuan\ProjectData\contact_rot.pkl')
    savePaths.append(R'H:\YangYuan\ProjectData\contact_final.pkl')
    Tm = transPlane(
        temRotPaths=pklPaths,
        savePaths=savePaths,
        mode='smplx',
    )
    print(Rotm)
    print(Tm)

def Unitest_SmplxRot():
    '''
    测试smplx是不是绕joint[0]旋转的
    测试成功，smplx旋转确实是绕joint[0]旋转的
    '''
    import torch
    import smplx
    import pickle as pkl
    pklPath = R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000n.pkl'
    with open(pklPath, 'rb') as file:
        data = pkl.load(file)
    model = smplx.create('./data/smplData/body_models', 'smplx',
                        gender='male', use_face_contour=False,
                        num_betas=10,
                        num_pca_comps=12,
                        ext='pkl')
    ## 不旋转
    output = model(
            betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
            global_orient = torch.tensor(np.zeros((1,3)).astype(np.float32)),
            body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
            left_hand_pose = torch.tensor(data['person00']['left_hand_pose'][None,:].astype(np.float32)),
            right_hand_pose = torch.tensor(data['person00']['right_hand_pose'][None,:].astype(np.float32)),
            transl = torch.tensor(np.zeros((1,3)).astype(np.float32)),
            jaw_pose = torch.tensor(data['person00']['jaw_pose'][None,:].astype(np.float32)))
    js = output.joints.detach().cpu().numpy().squeeze()
    meshData = MeshData()
    meshData.vert = output.vertices.detach().cpu().numpy().squeeze()
    meshData.face = model.faces + 1
    write_obj('notRot.obj',meshData)
    # 手动绕joint[0]旋转
    from scipy.spatial.transform import Rotation as R
    meshData.vert = R.from_rotvec(data['person00']['global_orient']).apply(meshData.vert-js[0][None,:]) + js[0][None,:] + data['person00']['transl'][None,:]
    write_obj('yyRot.obj',meshData)
    # smplx自身旋转
    outputR = model(
            betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
            global_orient = torch.tensor(data['person00']['global_orient'][None,:].astype(np.float32)),
            body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
            left_hand_pose = torch.tensor(data['person00']['left_hand_pose'][None,:].astype(np.float32)),
            right_hand_pose = torch.tensor(data['person00']['right_hand_pose'][None,:].astype(np.float32)),
            transl = torch.tensor(data['person00']['transl'][None,:].astype(np.float32)),
            jaw_pose = torch.tensor(data['person00']['jaw_pose'][None,:].astype(np.float32)))
    meshData.vert = outputR.vertices.detach().cpu().numpy().squeeze()
    meshData.face = model.faces + 1
    write_obj('smplxRot.obj',meshData)

def update_smplx_pkl():
    with open(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\vicon2scene.json', 'rb') as file:
        cam = np.array(json.load(file))
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            ## 测试手动给变换旋转
            ## 测试成功，变换能场景对齐
            # vs,js,fs=pkl2smpl(
            #     pklPath=os.path.join(frameDir,'000n.pkl'),
            #     mode='smplx'
            # )
            # vs = Camera_project(vs,cam)
            # meshData = MeshData()
            # meshData.vert = vs
            # meshData.face = fs
            # write_obj(os.path.join(frameDir,'000_scene_rot.obj'),meshData)
            
            applyRot2Smpl(
                pklPath=os.path.join(frameDir,'000n.pkl'),
                savePath=os.path.join(frameDir,'000_scene.pkl'),
                Rotm=cam[:3,:3],
                Tm=cam[:3,3],
                mode='smplx'
            )
            pkl2smpl(
                pklPath=os.path.join(frameDir,'000_scene.pkl'),
                mode='smplx',
                savePath=os.path.join(frameDir,'000_scene.obj'),
            )

def TestResult():
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            pkl2smpl(
                os.path.join(frameDir,'smplx','000_final.pkl'),
                mode='smplx',
                savePath=os.path.join(frameDir,'smplx','000_final.obj'),
            )

if __name__ == '__main__':

    with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\cam2world\vicon.json', 'rb') as file:
        cam = json.load(file)
    cam1 = np.array([
        [0.99947102, 0.02250866, 0.02347415, 1.340276],
        [-0.02250866, -0.04222833, 0.99885441, -0.00790659],
        [0.02347415, -0.99885441, -0.04169936, -0.84238],
        [0,0,0,1]
    ])
    cam = np.dot(cam1,cam)
    applyRot2Smpl(
        R'H:\YangYuan\ProjectData\RGB_n.pkl',
        R'H:\YangYuan\ProjectData\RGB_1.pkl',
        cam[:3,:3],
        cam[:3,3],
        'smplx',
        'male'
    )
    pkl2smpl(
        R'H:\YangYuan\ProjectData\RGB_1.pkl',
        'smplx',
        gender='male',
        savePath=R'H:\YangYuan\ProjectData\test.obj'
    )
    print(1)


    # Unitest_FitPlaneSmpl()

    # pkl2smpl(
    #     R'H:\YangYuan\ProjectData\contact_1.pkl',
    #     'smplx',
    #     gender='male',
    #     savePath=R'H:\YangYuan\ProjectData\test.obj'
    # )

    # cam = np.array([
    #     [0.99947102, 0.02250866, 0.02347415, 1.340276],
    #     [-0.02250866, -0.04222833, 0.99885441, -0.00790659],
    #     [0.02347415, -0.99885441, -0.04169936, -0.84238],
    #     [0,0,0,1]
    # ])

    # applyRot2Smpl(
    #     R'H:\YangYuan\ProjectData\contact_duiqi.pkl',
    #     R'H:\YangYuan\ProjectData\contact_1.pkl',
    #     cam[:3,:3],
    #     cam[:3,3],
    #     'smplx',
    #     'male'
    # )
    # print(1)

    ## 场景对齐
    # with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\cam2world\vicon.json', 'rb') as file:
    #     cam = json.load(file)
    # applyRot2Smpl(
    #     R'H:\YangYuan\ProjectData\contact_n.pkl',
    #     R'H:\YangYuan\ProjectData\contact_duiqi.pkl',
    #     np.array(cam)[:3,:3],
    #     np.array(cam)[:3,3],
    #     'smplx',
    #     'male'
    # )

    # pkl2smpl(
    #     R'H:\YangYuan\ProjectData\contact_duiqi.pkl',
    #     'smplx',
    #     gender='male',
    #     ext='npz',
    #     savePath=R'H:\YangYuan\ProjectData\test.obj'
    # )
    # vs,js,fs = pkl2smpl(
    #     R'H:\YangYuan\ProjectData\contact_final.pkl',
    #     'smplx',
    #     gender='male',
    #     ext='npz'
    # )
    # with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\cam2world\vicon.json', 'rb') as file:
    #     cam = json.load(file)
    # vs = Camera_project(vs,cam)
    # meshData = read_obj(R'H:\YangYuan\ProjectData\000_w.obj')
    # meshData.vert = vs
    # write_obj(R'H:\YangYuan\ProjectData\test.obj',meshData)


    # import pickle as pkl
    # with open(R'H:\YangYuan\ProjectData\RGB.pkl', 'rb') as file:
    #     data = pkl.load(file)
    # temdata = {}
    # temdata['person00'] = {}
    # temdata['person00']['betas'] = data['betas']
    # temdata['person00']['body_pose'] = data['body_pose'][0]
    # temdata['person00']['jaw_pose'] = data['jaw_pose'][0]
    # temdata['person00']['right_hand_pose'] = data['right_hand_pose'][0]
    # temdata['person00']['left_hand_pose'] = data['left_hand_pose'][0]
    # temdata['person00']['global_orient'] = data['global_orient'][0]
    # temdata['person00']['transl'] = data['transl'][0]
    # temdata['person00']['num_pca_comps'] = 12
    # temdata['person00']['cam_intrinsic'] = []
    # temdata['person00']['cam_extrinsic'] = []
    # temdata['person00']['reye_pose'] = data['reye_pose'][0]
    # temdata['person00']['leye_pose'] = data['leye_pose'][0]
    # temdata['person00']['expression'] = data['expression'][0]
    # with open(R'H:\YangYuan\ProjectData\RGB_n.pkl', 'wb') as file:
    #     pkl.dump(temdata, file)
    # print(1)



    # pkl2smpl(
    #     R'H:\YangYuan\ProjectData\contact_final.pkl',
    #     'smplx',
    #     gender='male',
    #     savePath=R'H:\YangYuan\ProjectData\contact_final.obj'
    # )

    # Unitest_FitPlaneSmpl_Frame()

    # Unitest_SmplxRot()
    # update_smplx_pkl()
    # Unitest_FitPlaneSmpl()
    # TestResult()