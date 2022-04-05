import numpy as np
import pickle as pkl
import json
import os
import glob
import sys
sys.path.append('./')
from utils.smpl_utils import pkl2smpl,applyRot2Smpl,smplx2smpl
from utils.obj_utils import write_obj

def transProx2vcl(pklPath,savePath,smplPath):
    temPath = R'tem.pkl'
    # resultPath = R'temR.pkl'
    with open(pklPath, 'rb') as file:
        data = pkl.load(file)
    temdata = {}
    temdata['person00'] = {}
    temdata['person00']['betas'] = data['betas']
    temdata['person00']['body_pose'] = data['body_pose'][0]
    temdata['person00']['jaw_pose'] = data['jaw_pose'][0]
    temdata['person00']['right_hand_pose'] = data['right_hand_pose'][0]
    temdata['person00']['left_hand_pose'] = data['left_hand_pose'][0]
    temdata['person00']['global_orient'] = data['global_orient'][0]
    temdata['person00']['transl'] = data['transl'][0]
    temdata['person00']['num_pca_comps'] = 12
    temdata['person00']['cam_intrinsic'] = []
    temdata['person00']['cam_extrinsic'] = []
    temdata['person00']['reye_pose'] = data['reye_pose'][0]
    temdata['person00']['leye_pose'] = data['leye_pose'][0]
    temdata['person00']['expression'] = data['expression'][0]
    with open(temPath, 'wb') as file:
        pkl.dump(temdata, file)

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
        temPath,
        os.path.join(savePath,'000_smplx_vcl.pkl'),
        cam[:3,:3],
        cam[:3,3],
        'smplx',
        'male'
    )
    pkl2smpl(
        os.path.join(savePath,'000_smplx_vcl.pkl'),
        'smplx',
        gender='male',
        savePath=os.path.join(savePath,'000.obj')
    )

    smplx2smpl(
        # os.path.join(savePath,'000.obj'),
        savePath,
        os.path.join(smplPath,'000_smpl_vcl.pkl'))
    
    pkl2smpl(
        os.path.join(smplPath,'000_smpl_vcl.pkl'),
        'smpl',
        gender='male',
        savePath=os.path.join(smplPath,'000.obj')
    )

    os.remove(temPath)

if __name__ == '__main__':
    # with open(R'\\105.1.1.2\Body\NIPS2022_PhysContact\prox_quantiative_dataset\prox_results\rgb_contact\smpl\vicon_03301_02\s001_frame_00001__00.00.00.030\000_smpl_vcl.pkl', 'rb') as file:
    #     data = pkl.load(file)

    # savePath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\prox_quantiative_dataset\prox_results\gt'
    # path = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh'
    # import shutil
    # for seq in glob.glob(os.path.join(path,'*')):
    #     seqName = os.path.basename(seq)
    #     for frame in glob.glob(os.path.join(seq,'results','*')):
    #         frameName = os.path.basename(frame)
    #         os.makedirs(
    #             os.path.join(savePath,'smpl',seqName,frameName),
    #             exist_ok=True
    #         )
    #         os.makedirs(
    #             os.path.join(savePath,'smplx',seqName,frameName),
    #             exist_ok=True
    #         )
    #         shutil.copyfile(
    #             os.path.join(frame,'smpl','000_smpl_vcl.pkl'),
    #             os.path.join(savePath,'smpl',seqName,frameName,'000_smpl_vcl.pkl')
    #         )
    #         shutil.copyfile(
    #             os.path.join(frame,'smplx','000_vcl.pkl'),
    #             os.path.join(savePath,'smplx',seqName,frameName,'000_smplx_vcl.pkl')
    #         )

    if True:
        import glob
        pklPath = []
        pklPaths = R'H:\YangYuan\Code\phy_program\CodeBase\data\proxResult\PROX_results_contact_all'
        savePath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\results\ICCV21_PROX\prox_quantiative\smplx'
        savePathsmpl = R'\\105.1.1.2\Body\NIPS2022_PhysContact\results\ICCV21_PROX\prox_quantiative\smpl'
        
        for seq in glob.glob(os.path.join(pklPaths,'*'))[3:]:
            seqName = os.path.basename(seq)
            for frame in glob.glob(os.path.join(seq,'results','*')):
                os.makedirs(os.path.join(savePath,seqName,os.path.basename(frame)),exist_ok=True)
                os.makedirs(os.path.join(savePathsmpl,seqName,os.path.basename(frame)),exist_ok=True)
                transProx2vcl(
                    os.path.join(frame,'000.pkl'),
                    os.path.join(savePath,seqName,os.path.basename(frame)),
                    os.path.join(savePathsmpl,seqName,os.path.basename(frame))
                )

    # if True:
    #     import glob
    #     pklPath = []
    #     pklPath = R'H:\YangYuan\Code\phy_program\CodeBase\data\proxResult\PROX_results_contact_all'
    #     savePath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\prox_quantiative_dataset\prox_results\rgb\smplx\vicon_03301_03'
    #     savePathsmpl = R'\\105.1.1.2\Body\NIPS2022_PhysContact\prox_quantiative_dataset\prox_results\rgb\smpl\vicon_03301_03'
    #     for frame in glob.glob(os.path.join(pklPath,'*')):
    #         os.makedirs(os.path.join(savePath,os.path.basename(frame)),exist_ok=True)
    #         os.makedirs(os.path.join(savePathsmpl,os.path.basename(frame)),exist_ok=True)
    #         transProx2vcl(
    #             os.path.join(frame,'000.pkl'),
    #             os.path.join(savePath,os.path.basename(frame)),
    #             os.path.join(savePathsmpl,os.path.basename(frame))
    #         )
    # else:
    #     with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smplx\000_vcl.pkl','rb') as file:
    #         tem = pkl.load(file)
    #     print(1)

    #     path = R'\\105.1.1.2\Body\NIPS2022_PhysContact\prox_quantiative_dataset\prox_results\rgb'

    #     for seq in glob.glob(os.path.join(path,'smpl','*')):
    #         for fram in glob.glob(os.path.join(seq,'*')):
    #             with open(os.path.join(fram,'000_smpl_vcl.pkl'),'rb') as file:
    #                 data = pkl.load(file)
    #             data['person00']['cam_extrinsic'] = tem['person00']['cam_extrinsic']
    #             data['person00']['cam_intrinsic'] = tem['person00']['cam_intrinsic']
    #             with open(os.path.join(fram,'000_smpl_vcl.pkl'),'wb') as file:
    #                 pkl.dump(data,file)
    #     for seq in glob.glob(os.path.join(path,'smplx','*')):
    #         for fram in glob.glob(os.path.join(seq,'*')):
    #             with open(os.path.join(fram,'000_smplx_vcl.pkl'),'rb') as file:
    #                 data = pkl.load(file)
    #             data['person00']['cam_extrinsic'] = tem['person00']['cam_extrinsic']
    #             data['person00']['cam_intrinsic'] = tem['person00']['cam_intrinsic']
    #             with open(os.path.join(fram,'000_smplx_vcl.pkl'),'wb') as file:
    #                 pkl.dump(data,file)

# if __name__ == '__main__':
#     import glob
#     pklPath = []
#     for frame in glob.glob(os.path.join(R'H:\YangYuan\ProjectData\proxData\pkl','*')):
#         transProx2vcl(
#             os.path.join(frame,'000.pkl'),
#             os.path.join(R'H:\YangYuan\ProjectData\proxData\mesh',os.path.basename(frame)[:4]+'.obj')
#         )

# if __name__ == '__main__':
#     imgPath = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\recordings\vicon_03301_02\img\s001_frame_00001__00.00.00.030.jpg'
#     keyPath = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\keypoints\vicon_03301_02\s001_frame_00001__00.00.00.030_keypoints.json'
#     import json
#     with open(keyPath, 'rb') as file:
#         data = json.load(file)
#     keyPoints = data['people'][0]['pose_keypoints_2d']
#     import cv2
#     img = cv2.imread(imgPath)
#     img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
#     for point in keyPoints:
#         img = cv2.circle(img,(int(point[0])//2,int(point[1])//2),2,(255,0,0),6)
#         cv2.imshow('1',img)
#         cv2.waitKey(0)