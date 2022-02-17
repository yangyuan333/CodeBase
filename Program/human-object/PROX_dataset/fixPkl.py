import pickle as pkl
import os
import glob
import numpy as np
import torch
import sys
sys.path.append('./')

if __name__ == '__main__':
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            with open(os.path.join(frameDir,'smpl','000_smpl.pkl'), 'rb') as file:
                data = pkl.load(file)
            dataTem = {}
            dataTem['person00'] = {}
            dataTem['person00']['betas'] = data['betas'].detach().cpu().numpy()
            dataTem['person00']['scale'] = np.array([1.0]).astype(np.float32)
            dataTem['person00']['global_orient'] = data['global_orient'][0][0].detach().cpu().numpy()
            dataTem['person00']['transl'] = data['transl'][0].detach().cpu().numpy()
            dataTem['person00']['body_pose'] = data['body_pose'].reshape(-1).detach().cpu().numpy()
            dataTem['person00']['pose'] = torch.cat((data['global_orient'].reshape(1,-1),data['body_pose'].reshape(1,-1)),1)[0].detach().cpu().numpy()
            with open(os.path.join(frameDir,'smpl','000_smpl_standard.pkl'), 'wb') as file:
                pkl.dump(dataTem, file)
            # config['config'].append(os.path.join(frameDir,'smpl','000_smpl_rot.pkl'))
