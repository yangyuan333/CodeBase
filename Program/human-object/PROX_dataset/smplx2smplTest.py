import os
import os.path as osp
import sys
import pickle
import glob
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm

from smplx import build_layer

from transfer_model.config import parse_args,read_yaml
from transfer_model.data import build_dataloader
from transfer_model.transfer_model import run_fitting
from transfer_model.utils import read_deformation_transfer, np_mesh_to_o3d

import yaml
import sys
sys.path.append('./')
from utils.smpl_utils import smplx2smpl,SMPLModel
from utils.rotate_utils import *
from utils.obj_utils import MeshData,read_obj,write_obj

def main(exp_cfg) -> None:

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    logger.add(
        lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
        colorize=True)

    ## 自定义
    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder
    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    data_obj_dict = build_dataloader(exp_cfg)

    dataloader = data_obj_dict['dataloader']

    for ii, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        var_dict = run_fitting(
            exp_cfg, batch, body_model, def_matrix, mask_ids)
        paths = batch['paths']

        for ii, path in enumerate(paths):
            _, fname = osp.split(path)

            output_path = osp.join(
                output_folder, f'{osp.splitext(fname)[0]}.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(var_dict, f)

            output_path = osp.join(
                output_folder, f'{osp.splitext(fname)[0]}.obj')
            mesh = np_mesh_to_o3d(
                var_dict['vertices'][ii], var_dict['faces'])
            o3d.io.write_triangle_mesh(output_path, mesh)

if __name__ == '__main__':

    # with open(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\mesh\000_smplx_smpl.pkl'), 'rb') as file:
    #     data = pickle.load(file)
    #     pose = torch.cat((data['global_orient'].reshape(1,-1),data['body_pose'].reshape(1,-1)),1)
    #     beta = data['betas']
    #     transl = data['transl']
    #     smplModel = SMPLModel()
    #     vs, js = smplModel(
    #         beta.cpu(),
    #         pose.cpu(),
    #         transl.cpu(),
    #         torch.tensor([[1.0]])
    #     )
    #     meshData = read_obj('./data/smpl/template.obj')
    #     meshData.vert = vs[0].detach().numpy()
    #     write_obj(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\mesh\000_smplx_smpl.obj', meshData)
    #     meshData.vert = data['vertices'][0].detach().cpu().numpy()
    #     write_obj(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\mesh\000_smplx_smpl_1.obj', meshData)

    cfg = read_yaml(os.path.join(R'H:\YangYuan\Code\phy_program\CodeBase\data','smplx2smpl.yaml'))
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            cfg.datasets.mesh_folder.data_folder = os.path.join(frameDir,'mesh')
            cfg.output_folder = os.path.join(frameDir,'smpl')
            smplx2smpl(cfg)