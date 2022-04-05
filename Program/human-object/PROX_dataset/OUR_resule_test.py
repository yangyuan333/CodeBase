import sys
sys.path.append('./')
import numpy as np
import os
import glob

from utils.smpl_utils import pkl2smpl,applyRot2Smpl,smplx2smpl
from utils.obj_utils import write_obj
import pickle as pkl
import shutil

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

    shutil.copyfile(temPath,os.path.join(savePath,'000_smplx_vcl.pkl'))

    pkl2smpl(
        os.path.join(savePath,'000_smplx_vcl.pkl'),
        'smplx',
        gender='male',
        savePath=os.path.join(savePath,'000.obj')
    )

    smplx2smpl(
        savePath,
        os.path.join(smplPath,'000_smpl_vcl.pkl'))
    
    pkl2smpl(
        os.path.join(smplPath,'000_smpl_vcl.pkl'),
        'smpl',
        gender='male',
        savePath=os.path.join(smplPath,'000.obj')
    )

if __name__ == '__main__':
    ourPath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\results\Ours_kinematics\prox_quantiative'
    proxPath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\results\ICCV21_PROX\prox_quantiative'
    
    kinds = ['smpl','smplx']

    for kind in kinds:
        for seq in glob.glob(os.path.join(proxPath,kind,'*')):
            seqName = os.path.basename(seq)
            oursNames = glob.glob(os.path.join(ourPath,kind,seqName,'*'))
            idx = -1
            for frame in glob.glob(os.path.join(seq,'*')):
                idx += 1
                name = os.path.basename(frame)
                os.rename(
                    os.path.join(ourPath,kind,seqName,os.path.basename(oursNames[idx])),
                    os.path.join(ourPath,kind,seqName,name))


## add camera
# if __name__ == '__main__':
#     rootPath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\results\ICCV21_PROX\prox_quantiative'
#     with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smplx\000_vcl.pkl','rb') as file:
#         tem = pkl.load(file)
#     for seq in glob.glob(os.path.join(rootPath,'smplx','*')):
#         for frame in glob.glob(os.path.join(seq,'*')):
#             with open(os.path.join(frame,'000_smplx_vcl.pkl'),'rb') as file:
#                 data = pkl.load(file)
#             data['person00']['cam_extrinsic'] = tem['person00']['cam_extrinsic']
#             data['person00']['cam_intrinsic'] = tem['person00']['cam_intrinsic']
#             with open(os.path.join(frame,'000_smplx_vcl.pkl'),'wb') as file:
#                 pkl.dump(data,file)
#     for seq in glob.glob(os.path.join(rootPath,'smpl','*')):
#         for frame in glob.glob(os.path.join(seq,'*')):
#             with open(os.path.join(frame,'000_smpl_vcl.pkl'),'rb') as file:
#                 data = pkl.load(file)
#             data['person00']['cam_extrinsic'] = tem['person00']['cam_extrinsic']
#             data['person00']['cam_intrinsic'] = tem['person00']['cam_intrinsic']
#             with open(os.path.join(frame,'000_smpl_vcl.pkl'),'wb') as file:
#                 pkl.dump(data,file)

## trans result and to smpl
# if __name__ == '__main__':
#     path = R'H:\YangYuan\Code\phy_program\MvSMPLfitting\output_Linear_hand_face_contact_sdf_7_check_10_5_1e4_opAll\results'
#     savePath = R'\\105.1.1.2\Body\NIPS2022_PhysContact\results\Ours_kinematics\prox_quantiative'
#     with open(R'H:\YangYuan\Code\phy_program\MvSMPLfitting\output_Linear_hand_face_contact_sdf_7_check_10_5_1e4_opAll\results\vicon_03301_01_s001\00001\000.pkl','rb') as file:
#         data = pkl.load(file)
#     for framePath in glob.glob(os.path.join(path,'*')):
#         pklPath = os.path.join(framePath,'00001','000.pkl')
#         seqName = os.path.basename(framePath)[:14]
#         frameName = os.path.basename(framePath)[15:]
#         os.makedirs(os.path.join(savePath,'smplx',seqName,frameName),exist_ok=True)
#         os.makedirs(os.path.join(savePath,'smpl',seqName,frameName),exist_ok=True)
#         transProx2vcl(
#             pklPath,
#             os.path.join(savePath,'smplx',seqName,frameName),
#             os.path.join(savePath,'smpl',seqName,frameName)
#         )
#         print(1)