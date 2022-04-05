import pickle as pkl
import json
import sys
sys.path.append('./')
from utils.smpl_utils import pkl2smpl,applyRot2Smpl
from utils.obj_utils import MeshData,write_obj

if __name__ == '__main__':
    with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_qualitative_dataset\cam2world\BasementSittingBooth.json','rb') as file:
        cam = json.load(file)
    applyRot2Smpl(
        R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_qualitative_dataset\PROXD Fittings\PROXD\BasementSittingBooth_00142_01\results\s001_frame_00001__00.00.00.029\000_n.pkl',
        
    )
    # import glob
    # import os
    # rootPath = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_qualitative_dataset\PROXD Fittings\PROXD'
    # for squenceId in glob.glob(os.path.join(rootPath,'*')):
    #     for frameId in glob.glob(os.path.join(squenceId,'results','*')):
    #         if os.path.exists(os.path.join(frameId,'000.pkl')):
    #             with open(os.path.join(frameId,'000.pkl'), 'rb') as file:
    #                 data = pkl.load(file)
    #             temdata = {}
    #             temdata['person00'] = {}
    #             temdata['person00']['betas'] = data['betas']
    #             temdata['person00']['body_pose'] = data['body_pose'][0]
    #             temdata['person00']['jaw_pose'] = data['jaw_pose'][0]
    #             temdata['person00']['right_hand_pose'] = data['right_hand_pose'][0]
    #             temdata['person00']['left_hand_pose'] = data['left_hand_pose'][0]
    #             temdata['person00']['global_orient'] = data['global_orient'][0]
    #             temdata['person00']['transl'] = data['transl'][0]
    #             temdata['person00']['num_pca_comps'] = 12
    #             temdata['person00']['cam_intrinsic'] = []
    #             temdata['person00']['cam_extrinsic'] = []
    #             temdata['person00']['reye_pose'] = data['reye_pose'][0]
    #             temdata['person00']['leye_pose'] = data['leye_pose'][0]
    #             temdata['person00']['expression'] = data['expression'][0]
    #             with open(os.path.join(frameId,'000_n.pkl'), 'wb') as file1:
    #                 pkl.dump(temdata,file1)