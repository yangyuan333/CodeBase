import os.path as osp
import os
import argparse
import pickle as pkl
import numpy as np
import torch
import json
import glob
import sys
sys.path.append('./')
import smplx
from utils.obj_utils import MeshData, write_obj
from utils.rotate_utils import *
from utils.smpl_utils import smplxMain

def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    print(model)

    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)
    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.show()
    elif plotting_module == 'open3d':
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)
    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))

# def smplxMain(config):
#     with open(config['pklPath'], 'rb') as file:
#         data = pkl.load(file, encoding='iso-8859-1')
        
#         if 'num_pca_comps' in data:
#             config['num_pca_comps'] = data['num_pca_comps']
#         if 'num_betas' in data:
#             config['num_betas'] = data['num_betas']
#         if 'gender' in data:
#             config['gender'] = data['gender']

#         model = smplx.create(config['modelPath'], 'smplx',
#                             gender=config['gender'], use_face_contour=False,
#                             num_betas=config['num_betas'],
#                             num_pca_comps=config['num_pca_comps'],
#                             ext=config['ext'])

#         output = model(
#             betas = torch.tensor(data['beta'][None,:]),
#             global_orient = torch.tensor(data['global_orient']),
#             body_pose = torch.tensor(data['body_pose']),
#             left_hand_pose = torch.tensor(data['left_hand_pose']),
#             right_hand_pose = torch.tensor(data['right_hand_pose']),
#             transl = torch.tensor(data['transl']),
#             jaw_pose = torch.tensor(data['jaw_pose']),
#             return_verts = True,
#         )
#         vertices = output.vertices.detach().cpu().numpy().squeeze()
#         joints = output.joints.detach().cpu().numpy().squeeze()

#         if ('savePath' in config) and (config['savePath'] != ''):
#             meshData = MeshData()
#             meshData.vert = vertices
#             meshData.face = model.faces + 1
#             write_obj(config['savePath'], meshData)
#             return vertices, joints
#         return vertices, joints

#         with open(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\vicon2scene.json', 'rb') as file:
#             cam = np.array(json.load(file))
#         meshData.vert = Camera_project(meshData.vert, cam)

#         write_obj(config['savePath'], meshData)

if __name__ == '__main__':
    # config = {
    #     'modelPath' : R'H:\YangYuan\Code\phy_program\CodeBase\data\models_smplx_v1_1\models',
    #     'pklPath'   : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000.pkl',
    #     'savePath'  : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000x.obj',
    #     'gender'    : 'male',
    #     'num_betas' : 10,
    #     'num_pca_comps' : 12,
    #     'ext'       : 'npz',
    # }
    # smplxTest(config)
    with open(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\vicon2scene.json', 'rb') as file:
        cam = np.array(json.load(file))
    config = {
        'modelPath' : R'H:\YangYuan\Code\phy_program\CodeBase\data\models_smplx_v1_1\models',
        'pklPath'   : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000.pkl',
        'savePath'  : R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\000x.obj',
        'gender'    : 'male',
        'num_betas' : 10,
        'num_pca_comps' : 12,
        'ext'       : 'npz',
        'body_only' : False,
    }
    for seqDir in glob.glob(os.path.join(R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh','*')):
        for frameDir in glob.glob(os.path.join(seqDir,'results','*')):
            config['pklPath'] = os.path.join(frameDir,'000.pkl')
            config['savePath'] = ''
            config['body_only'] = False
            vs,_,fs = smplxMain(config)

            os.makedirs(os.path.join(frameDir,'mesh'),exist_ok=True)

            savePath = os.path.join(frameDir,'mesh','000_smplx.obj')
            meshData = MeshData()
            meshData.vert = vs
            meshData.face = fs
            meshData.vert = Camera_project(meshData.vert, cam)
            write_obj(savePath, meshData)