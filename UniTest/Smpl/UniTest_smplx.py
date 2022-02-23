import smplx
import pickle as pkl
import torch
import sys
sys.path.append('./')
from utils.obj_utils import MeshData, write_obj
from utils.smpl_utils import pkl2Smpl

def Uni_smplx(pklPath, mode='smpl', modelPath='./data/smplData/body_models', gender='male', ext='pkl', **config):
    with open(pklPath, 'rb') as file:
        data = pkl.load(file, encoding='iso-8859-1')

        if mode.lower() == 'smpl':
            model = smplx.create(modelPath,mode,gender=gender)
            output = model(
                betas = torch.tensor(data['person00']['betas']),
                body_pose = torch.tensor(data['person00']['body_pose'][None,:]),
                global_orient = torch.tensor(data['person00']['global_orient'][None,:]),
                transl = torch.tensor(data['person00']['transl'][None,:]))
        elif mode.lower() == 'smplx':
            model = smplx.create(modelPath, mode,
                                gender=gender, use_face_contour=False,
                                num_betas=data['person00']['betas'].shape[1],
                                num_pca_comps=data['person00']['left_hand_pose'].shape[0],
                                ext=ext)
            output = model(
                    betas = torch.tensor(data['person00']['betas']),
                    global_orient = torch.tensor(data['person00']['global_orient'][None,:]),
                    body_pose = torch.tensor(data['person00']['body_pose'][None,:]),
                    left_hand_pose = torch.tensor(data['person00']['left_hand_pose'][None,:]),
                    right_hand_pose = torch.tensor(data['person00']['right_hand_pose'][None,:]),
                    transl = torch.tensor(data['person00']['transl'][None,:]),
                    jaw_pose = torch.tensor(data['person00']['jaw_pose'][None,:]))
    
    if 'savePath' in config:
        meshData = MeshData()
        meshData.vert = output.vertices.detach().cpu().numpy().squeeze()
        meshData.face = model.faces + 1
        write_obj(config['savePath'], meshData)
    return output.vertices.detach().cpu().numpy().squeeze(),output.joints.detach().cpu().numpy().squeeze(),model.faces + 1

if __name__ == '__main__':
    vs, js, fs = Uni_smplx(
        R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smpl\000_smpl_final.pkl',
        savePath=R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smpl\test.obj'
    )

    ## 两种smpl节点测试
    meshData = MeshData()
    meshData.vert = js[:24,:]
    write_obj('js_test.obj',meshData)
    vs1,js1 = pkl2Smpl(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smpl\000_smpl_final.pkl')
    meshData = MeshData()
    meshData.vert = js1
    write_obj('js_test1.obj',meshData)
    ## 两种smpl节点测试
    
    Uni_smplx(
        R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000n.pkl',
        mode='smplx',
        savePath=R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\test.obj'
    )
