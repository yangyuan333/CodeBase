""" 按q刷新smpl和humanoid

"""

import sys
sys.path.append('./')

from numpy.core.numeric import full
from numpy.lib.function_base import angle

import os
import math
import torch
import numpy as np
import pybullet as p
import pybullet_data

from smplx import SMPL
from RL.src.data_utils.math_utils import rvec2mat, mat2joint, mat2rvec, mat2pb, mat2quat, quat2euler, rvec2axisAndAngle

# 可以控制的关节 全部为3自由度的spherical类型关节
chest = 1
neck = 2
rightHip = 3
rightKnee = 4     
rightAnkle = 5
rightShoulder = 6
rightElbow = 7    
rightWrist = 8  
leftHip = 9
leftKnee = 10     
leftAnkle = 11
leftShoulder = 12
leftElbow = 13     
leftWrist = 14  

chest_inSMPL = 9
neck_inSMPL = 12
rightHip_inSMPL = 2
rightKnee_inSMPL = 5
rightAnkle_inSMPL = 8
rightShoulder_inSMPL = 17
rightElbow_inSMPL = 19   
rightWrist_inSMPL = 21    
leftHip_inSMPL = 1
leftKnee_inSMPL = 4  
leftAnkle_inSMPL = 7
leftShoulder_inSMPL = 16
leftElbow_inSMPL = 18
leftWrist_inSMPL = 20

SMPL_POSE_SIZE = 72

def save_mesh_obj(filename, vertices, faces):
    """Save a mesh as an obj file.

    Arguments:
        filename {str} -- output file name
        vertices {array} -- mesh vertices
        faces {array} -- mesh faces
    """
    with open(filename, 'w') as obj:
        for vert in vertices:
            obj.write('v {:f} {:f} {:f}\n'.format(*vert))
        for face in faces + 1:  # Faces are 1-based, not 0-based in obj files
            obj.write('f {:d} {:d} {:d}\n'.format(*face))

def main(**args):

    # 读取参数
    useYup = args.get('useYUp')
    cameraArgs = {
        'cameraDistance': args.get('cameraDistance'),
        'cameraYaw': args.get('cameraYaw'),
        'cameraPitch': args.get('cameraPitch'),
        'cameraTargetPosition': args.get('cameraTargetPosition')
    }
    humanoidArgs = {
        'urdf' : args.get('urdf')
    }
    data_folder = args.get('data_folder')
    smpl_mesh_path = args.get('smpl_mesh_path')
    save_mesh_root = args.get('save_mesh_root')
    smpl_model_root = args.get('smpl_model_root')

    p.connect(p.GUI)
    if useYup:
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.setGravity(0, -9.8, 0)
    p.resetDebugVisualizerCamera(**cameraArgs)

    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # 加载urdf
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0]) # 绕x轴旋转-90度
    p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)

    humanoid = p.loadURDF(
        fileName = humanoidArgs.get('urdf'),
        basePosition = [2, 1, 0],
        globalScaling = 1.0,
        useFixedBase = True,
        flags=p.URDF_MAINTAIN_LINK_ORDER
    )

    # 加载SMPL T-pose mesh
    meshscale = [1, 1, 1]
    baseposition = [0, 1.2, 0]
    color = [0.9, 0.7, 0.7, 1]
    smpl_visual_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                         fileName=smpl_mesh_path,
                                         meshScale = meshscale,
                                         rgbaColor = color)
    smpl_collision_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName=smpl_mesh_path,
                                            meshScale = meshscale)
    smpl = p.createMultiBody(baseCollisionShapeIndex=smpl_collision_id,
                                            baseVisualShapeIndex=smpl_visual_id,
                                            basePosition=baseposition)

    # SMPL前向传播，保存obj，更新visualShape
    SMPL_layer = SMPL(model_path=smpl_model_root, batch_size=1)
    smpl_output = SMPL_layer()
    vertices = smpl_output.vertices[0].detach().numpy() # 6890 x 3
    faces = SMPL_layer.faces # 13776 x 3
    save_mesh_obj(os.path.join(save_mesh_root, 'test_Tpose.obj'), vertices, faces)

    # base debug
    # rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
    # pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, 0)
    # yawId = p.addUserDebugParameter("yaw", -3.14, 3.14, 0)
    # translxId = p.addUserDebugParameter("transl_x", -1, 1, 0)
    # translyId = p.addUserDebugParameter("transl_y", -1, 1, 0)
    # translzId = p.addUserDebugParameter("transl_z", -1, 1, 0)

    # left hip pose vector
    # axis-angle 表示
    fullPose = torch.zeros((1, SMPL_POSE_SIZE), dtype=torch.float32)
    rot_axis_x = np.array([1, 0, 0], dtype=np.float) # 转轴 世界坐标系的x轴
    rot_axis_y = np.array([0, 1, 0], dtype=np.float) # 转轴 世界坐标系的y轴
    rot_axis_z = np.array([0, 0, 1], dtype=np.float) # z轴正方向为旋转轴

    smpl_chest_x_id = p.addUserDebugParameter("smpl_chest_x", -3.14, 3.14, 0)
    smpl_chest_y_id = p.addUserDebugParameter("smpl_chest_y", -3.14, 3.14, 0)
    smpl_chest_z_id = p.addUserDebugParameter("smpl_chest_z", -3.14, 3.14, 0)

    smpl_neck_x_id = p.addUserDebugParameter("smpl_neck_x", -3.14, 3.14, 0)
    smpl_neck_y_id = p.addUserDebugParameter("smpl_neck_y", -3.14, 3.14, 0)
    smpl_neck_z_id = p.addUserDebugParameter("smpl_neck_z", -3.14, 3.14, 0)

    smpl_rightHip_x_id = p.addUserDebugParameter("smpl_rightHip_x", -3.14, 3.14, 0)
    smpl_rightHip_y_id = p.addUserDebugParameter("smpl_rightHip_y", -3.14, 3.14, 0)
    smpl_rightHip_z_id = p.addUserDebugParameter("smpl_rightHip_z", -3.14, 3.14, 0)

    smpl_rightKnee_x_id = p.addUserDebugParameter("smpl_rightKnee_x", -3.14, 3.14, 0)
    smpl_rightKnee_y_id = p.addUserDebugParameter("smpl_rightKnee_y", -3.14, 3.14, 0)
    smpl_rightKnee_z_id = p.addUserDebugParameter("smpl_rightKnee_z", -3.14, 3.14, 0)

    smpl_rightAnkle_x_id = p.addUserDebugParameter("smpl_rightAnkle_x", -3.14, 3.14, 0)
    smpl_rightAnkle_y_id = p.addUserDebugParameter("smpl_rightAnkle_y", -3.14, 3.14, 0)
    smpl_rightAnkle_z_id = p.addUserDebugParameter("smpl_rightAnkle_z", -3.14, 3.14, 0)

    smpl_rightShoulder_x_id = p.addUserDebugParameter("smpl_rightShoulder_x", -3.14, 3.14, 0)
    smpl_rightShoulder_y_id = p.addUserDebugParameter("smpl_rightShoulder_y", -3.14, 3.14, 0)
    smpl_rightShoulder_z_id = p.addUserDebugParameter("smpl_rightShoulder_z", -3.14, 3.14, 0)

    smpl_rightElbow_x_id = p.addUserDebugParameter("smpl_rightElbow_x", -3.14, 3.14, 0)
    smpl_rightElbow_y_id = p.addUserDebugParameter("smpl_rightElbow_y", -3.14, 3.14, 0)
    smpl_rightElbow_z_id = p.addUserDebugParameter("smpl_rightElbow_z", -3.14, 3.14, 0)

    smpl_rightWrist_x_id = p.addUserDebugParameter("smpl_rightWrist_x", -3.14, 3.14, 0)
    smpl_rightWrist_y_id = p.addUserDebugParameter("smpl_rightWrist_y", -3.14, 3.14, 0)
    smpl_rightWrist_z_id = p.addUserDebugParameter("smpl_rightWrist_z", -3.14, 3.14, 0)

    smpl_leftHip_x_id = p.addUserDebugParameter("smpl_leftHip_x", -3.14, 3.14, 0)
    smpl_leftHip_y_id = p.addUserDebugParameter("smpl_leftHip_y", -3.14, 3.14, 0)
    smpl_leftHip_z_id = p.addUserDebugParameter("smpl_leftHip_z", -3.14, 3.14, 0)

    smpl_leftKnee_x_id = p.addUserDebugParameter("smpl_leftKnee_x", -3.14, 3.14, 0)
    smpl_leftKnee_y_id = p.addUserDebugParameter("smpl_leftKnee_y", -3.14, 3.14, 0)
    smpl_leftKnee_z_id = p.addUserDebugParameter("smpl_leftKnee_z", -3.14, 3.14, 0)

    smpl_leftAnkle_x_id = p.addUserDebugParameter("smpl_leftAnkle_x", -3.14, 3.14, 0)
    smpl_leftAnkle_y_id = p.addUserDebugParameter("smpl_leftAnkle_y", -3.14, 3.14, 0)
    smpl_leftAnkle_z_id = p.addUserDebugParameter("smpl_leftAnkle_z", -3.14, 3.14, 0)

    smpl_leftShoulder_x_id = p.addUserDebugParameter("smpl_leftShoulder_x", -3.14, 3.14, 0)
    smpl_leftShoulder_y_id = p.addUserDebugParameter("smpl_leftShoulder_y", -3.14, 3.14, 0)
    smpl_leftShoulder_z_id = p.addUserDebugParameter("smpl_leftShoulder_z", -3.14, 3.14, 0)

    smpl_leftElbow_x_id = p.addUserDebugParameter("smpl_leftElbow_x", -3.14, 3.14, 0)
    smpl_leftElbow_y_id = p.addUserDebugParameter("smpl_leftElbow_y", -3.14, 3.14, 0)
    smpl_leftElbow_z_id = p.addUserDebugParameter("smpl_leftElbow_z", -3.14, 3.14, 0)

    smpl_leftWrist_x_id = p.addUserDebugParameter("smpl_leftWrist_x", -3.14, 3.14, 0)
    smpl_leftWrist_y_id = p.addUserDebugParameter("smpl_leftWrist_y", -3.14, 3.14, 0)
    smpl_leftWrist_z_id = p.addUserDebugParameter("smpl_leftWrist_z", -3.14, 3.14, 0)

    frame_id = 0
    while(p.isConnected()):

        smpl_chest_x_angle = p.readUserDebugParameter(smpl_chest_x_id)
        smpl_chest_y_angle = p.readUserDebugParameter(smpl_chest_y_id)
        smpl_chest_z_angle = p.readUserDebugParameter(smpl_chest_z_id)

        smpl_neck_x_angle = p.readUserDebugParameter(smpl_neck_x_id)
        smpl_neck_y_angle = p.readUserDebugParameter(smpl_neck_y_id)
        smpl_neck_z_angle = p.readUserDebugParameter(smpl_neck_z_id)

        smpl_rightHip_x_angle = p.readUserDebugParameter(smpl_rightHip_x_id)
        smpl_rightHip_y_angle = p.readUserDebugParameter(smpl_rightHip_y_id)
        smpl_rightHip_z_angle = p.readUserDebugParameter(smpl_rightHip_z_id)

        smpl_rightKnee_x_angle = p.readUserDebugParameter(smpl_rightKnee_x_id)
        smpl_rightKnee_y_angle = p.readUserDebugParameter(smpl_rightKnee_y_id)
        smpl_rightKnee_z_angle = p.readUserDebugParameter(smpl_rightKnee_z_id)

        smpl_rightAnkle_x_angle = p.readUserDebugParameter(smpl_rightAnkle_x_id)
        smpl_rightAnkle_y_angle = p.readUserDebugParameter(smpl_rightAnkle_y_id)
        smpl_rightAnkle_z_angle = p.readUserDebugParameter(smpl_rightAnkle_z_id)

        smpl_rightShoulder_x_angle = p.readUserDebugParameter(smpl_rightShoulder_x_id)
        smpl_rightShoulder_y_angle = p.readUserDebugParameter(smpl_rightShoulder_y_id)
        smpl_rightShoulder_z_angle = p.readUserDebugParameter(smpl_rightShoulder_z_id)

        smpl_rightElbow_x_angle = p.readUserDebugParameter(smpl_rightElbow_x_id)
        smpl_rightElbow_y_angle = p.readUserDebugParameter(smpl_rightElbow_y_id)
        smpl_rightElbow_z_angle = p.readUserDebugParameter(smpl_rightElbow_z_id)

        smpl_rightWrist_x_angle = p.readUserDebugParameter(smpl_rightWrist_x_id)
        smpl_rightWrist_y_angle = p.readUserDebugParameter(smpl_rightWrist_y_id)
        smpl_rightWrist_z_angle = p.readUserDebugParameter(smpl_rightWrist_z_id)

        smpl_leftHip_x_angle = p.readUserDebugParameter(smpl_leftHip_x_id)
        smpl_leftHip_y_angle = p.readUserDebugParameter(smpl_leftHip_y_id)
        smpl_leftHip_z_angle = p.readUserDebugParameter(smpl_leftHip_z_id)

        smpl_leftKnee_x_angle = p.readUserDebugParameter(smpl_leftKnee_x_id)
        smpl_leftKnee_y_angle = p.readUserDebugParameter(smpl_leftKnee_y_id)
        smpl_leftKnee_z_angle = p.readUserDebugParameter(smpl_leftKnee_z_id)

        smpl_leftAnkle_x_angle = p.readUserDebugParameter(smpl_leftAnkle_x_id)
        smpl_leftAnkle_y_angle = p.readUserDebugParameter(smpl_leftAnkle_y_id)
        smpl_leftAnkle_z_angle = p.readUserDebugParameter(smpl_leftAnkle_z_id)

        smpl_leftShoulder_x_angle = p.readUserDebugParameter(smpl_leftShoulder_x_id)
        smpl_leftShoulder_y_angle = p.readUserDebugParameter(smpl_leftShoulder_y_id)
        smpl_leftShoulder_z_angle = p.readUserDebugParameter(smpl_leftShoulder_z_id)

        smpl_leftElbow_x_angle = p.readUserDebugParameter(smpl_leftElbow_x_id)
        smpl_leftElbow_y_angle = p.readUserDebugParameter(smpl_leftElbow_y_id)
        smpl_leftElbow_z_angle = p.readUserDebugParameter(smpl_leftElbow_z_id)

        smpl_leftWrist_x_angle = p.readUserDebugParameter(smpl_leftWrist_x_id)
        smpl_leftWrist_y_angle = p.readUserDebugParameter(smpl_leftWrist_y_id)
        smpl_leftWrist_z_angle = p.readUserDebugParameter(smpl_leftWrist_z_id)

        # base_roll = p.readUserDebugParameter(rollId)
        # base_pitch = p.readUserDebugParameter(pitchId)
        # base_yaw = p.readUserDebugParameter(yawId)
        # base_x = p.readUserDebugParameter(translxId)
        # base_y = p.readUserDebugParameter(translyId)
        # base_z = p.readUserDebugParameter(translzId)
        # base_orn = p.getQuaternionFromEuler([base_roll, base_pitch, base_yaw])
        # p.resetBasePositionAndOrientation(humanoid, [base_x, base_y, base_z], base_orn)

        qKey = ord('q')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
            # 检测到按下qKey建
            print('press [q]')

            # transl, full_body_pose = mocap_data.get_fullPose(real_frame_index)

            smpl_chest_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_chest_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_chest_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_chest_z_angle))

            smpl_neck_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_neck_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_neck_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_neck_z_angle))

            smpl_rightHip_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_rightHip_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_rightHip_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_rightHip_z_angle))

            smpl_rightKnee_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_rightKnee_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_rightKnee_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_rightKnee_z_angle))

            smpl_rightAnkle_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_rightAnkle_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_rightAnkle_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_rightAnkle_z_angle))

            smpl_rightShoulder_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_rightShoulder_x_angle) @
                                          rvec2mat(rot_axis_y * smpl_rightShoulder_y_angle) @
                                          rvec2mat(rot_axis_z * smpl_rightShoulder_z_angle))

            smpl_rightElbow_pose = mat2rvec(rvec2mat(rot_axis_z * smpl_rightElbow_z_angle) @
                                          rvec2mat(rot_axis_y * smpl_rightElbow_y_angle) @
                                          rvec2mat(rot_axis_x * smpl_rightElbow_x_angle))
            
            smpl_rightWrist_pose = mat2rvec(rvec2mat(rot_axis_z * smpl_rightWrist_z_angle) @
                                          rvec2mat(rot_axis_y * smpl_rightWrist_y_angle) @
                                          rvec2mat(rot_axis_x * smpl_rightWrist_x_angle))

            smpl_leftHip_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_leftHip_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_leftHip_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_leftHip_z_angle))

            smpl_leftKnee_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_leftKnee_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_leftKnee_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_leftKnee_z_angle))

            smpl_leftAnkle_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_leftAnkle_x_angle) @ 
                                          rvec2mat(rot_axis_y * smpl_leftAnkle_y_angle) @ 
                                          rvec2mat(rot_axis_z * smpl_leftAnkle_z_angle))

            smpl_leftShoulder_pose = mat2rvec(rvec2mat(rot_axis_x * smpl_leftShoulder_x_angle) @
                                          rvec2mat(rot_axis_y * smpl_leftShoulder_y_angle) @
                                          rvec2mat(rot_axis_z * smpl_leftShoulder_z_angle))

            smpl_leftElbow_pose = mat2rvec(rvec2mat(rot_axis_z * smpl_leftElbow_z_angle) @
                                          rvec2mat(rot_axis_y * smpl_leftElbow_y_angle) @
                                          rvec2mat(rot_axis_x * smpl_leftElbow_x_angle))

            smpl_leftWrist_pose = mat2rvec(rvec2mat(rot_axis_z * smpl_leftWrist_z_angle) @
                                          rvec2mat(rot_axis_y * smpl_leftWrist_y_angle) @
                                          rvec2mat(rot_axis_x * smpl_leftWrist_x_angle))

            fullPose[0, 3*chest_inSMPL:3*chest_inSMPL+3] = torch.tensor(smpl_chest_pose)
            fullPose[0, 3*neck_inSMPL:3*neck_inSMPL+3] = torch.tensor(smpl_neck_pose)
            fullPose[0, 3*rightHip_inSMPL:3*rightHip_inSMPL+3] = torch.tensor(smpl_rightHip_pose)
            fullPose[0, 3*rightKnee_inSMPL:3*rightKnee_inSMPL+3] = torch.tensor(smpl_rightKnee_pose)
            fullPose[0, 3*rightAnkle_inSMPL:3*rightAnkle_inSMPL+3] = torch.tensor(smpl_rightAnkle_pose)
            fullPose[0, 3*rightShoulder_inSMPL:3*rightShoulder_inSMPL+3] = torch.tensor(smpl_rightShoulder_pose)
            fullPose[0, 3*rightElbow_inSMPL:3*rightElbow_inSMPL+3] = torch.tensor(smpl_rightElbow_pose)
            fullPose[0, 3*rightWrist_inSMPL:3*rightWrist_inSMPL+3] = torch.tensor(smpl_rightWrist_pose)
            fullPose[0, 3*leftHip_inSMPL:3*leftHip_inSMPL+3] = torch.tensor(smpl_leftHip_pose)
            fullPose[0, 3*leftKnee_inSMPL:3*leftKnee_inSMPL+3] = torch.tensor(smpl_leftKnee_pose)
            fullPose[0, 3*leftAnkle_inSMPL:3*leftAnkle_inSMPL+3] = torch.tensor(smpl_leftAnkle_pose)
            fullPose[0, 3*leftShoulder_inSMPL:3*leftShoulder_inSMPL+3] = torch.tensor(smpl_leftShoulder_pose)
            fullPose[0, 3*leftElbow_inSMPL:3*leftElbow_inSMPL+3] = torch.tensor(smpl_leftElbow_pose)
            fullPose[0, 3*leftWrist_inSMPL:3*leftWrist_inSMPL+3] = torch.tensor(smpl_leftWrist_pose)

            # 前向传播，保存，更新visualShape
            smpl_output = SMPL_layer(body_pose=fullPose[:, 3:])
            vertices = smpl_output.vertices[0].detach().numpy() # 6890 x 3
            faces = SMPL_layer.faces # 13776 x 3

            path = os.path.join(save_mesh_root, f'test_frame_{frame_id}.obj')
            save_mesh_obj(path, vertices, faces)
            
            # 删除
            p.removeBody(smpl)

            meshscale = [1, 1, 1]
            baseposition = [0, 1.2, 0]
            color = [0.9, 0.7, 0.7, 1]

            smpl_visual_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=path,
                                                meshScale = meshscale,
                                                rgbaColor = color)
            smpl_collision_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                    fileName=path,
                                                    meshScale = meshscale)
            smpl = p.createMultiBody(baseCollisionShapeIndex=smpl_collision_id,
                                                    baseVisualShapeIndex=smpl_visual_id,
                                                    basePosition=baseposition)
            os.remove(path) # 删除保存的obj文件
            frame_id += 1

            # 更新humanoid的pose

            # ####### Chest
            smpl_chest_axis_angle = fullPose[0, 3*chest_inSMPL:3*chest_inSMPL+3].numpy()
            smpl_chest_world_axis, common_chest_angle = rvec2axisAndAngle(smpl_chest_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            humanoid_chest_local_axis = linear_transform_mat @ smpl_chest_world_axis.reshape(3, 1)
            humanoid_chest_axis_angle = humanoid_chest_local_axis * common_chest_angle
            p.resetJointStateMultiDof(humanoid, chest, mat2pb(rvec2mat(humanoid_chest_axis_angle)))

            # ####### neck
            smpl_neck_axis_angle = fullPose[0, 3*neck_inSMPL:3*neck_inSMPL+3].numpy()
            smpl_neck_world_axis, common_neck_angle = rvec2axisAndAngle(smpl_neck_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            humanoid_neck_local_axis = linear_transform_mat @ smpl_neck_world_axis.reshape(3, 1)
            humanoid_neck_axis_angle = humanoid_neck_local_axis * common_neck_angle
            p.resetJointStateMultiDof(humanoid, neck, mat2pb(rvec2mat(humanoid_neck_axis_angle)))

            # ####### rightHip
            smpl_rightHip_axis_angle = fullPose[0, 3*rightHip_inSMPL:3*rightHip_inSMPL+3].numpy()
            smpl_rightHip_world_axis, common_rightHip_angle = rvec2axisAndAngle(smpl_rightHip_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            humanoid_rightHip_local_axis = linear_transform_mat @ smpl_rightHip_world_axis.reshape(3, 1)
            humanoid_rightHip_axis_angle = humanoid_rightHip_local_axis * common_rightHip_angle
            p.resetJointStateMultiDof(humanoid, rightHip, mat2pb(rvec2mat(humanoid_rightHip_axis_angle)))

            # ####### rightKnee  
            smpl_rightKnee_axis_angle = fullPose[0, 3*rightKnee_inSMPL:3*rightKnee_inSMPL+3].numpy()
            smpl_rightKnee_world_axis, common_rightKnee_angle = rvec2axisAndAngle(smpl_rightKnee_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            humanoid_rightKnee_local_axis = linear_transform_mat @ smpl_rightKnee_world_axis.reshape(3, 1)
            humanoid_rightKnee_axis_angle = humanoid_rightKnee_local_axis * common_rightKnee_angle
            p.resetJointStateMultiDof(humanoid, rightKnee, mat2pb(rvec2mat(humanoid_rightKnee_axis_angle)))

            # ####### rightAnkle
            smpl_rightAnkle_axis_angle = fullPose[0, 3*rightAnkle_inSMPL:3*rightAnkle_inSMPL+3].numpy()
            smpl_rightAnkle_world_axis, common_rightAnkle_angle = rvec2axisAndAngle(smpl_rightAnkle_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            humanoid_rightAnkle_local_axis = linear_transform_mat @ smpl_rightAnkle_world_axis.reshape(3, 1)
            humanoid_rightAnkle_axis_angle = humanoid_rightAnkle_local_axis * common_rightAnkle_angle
            p.resetJointStateMultiDof(humanoid, rightAnkle, mat2pb(rvec2mat(humanoid_rightAnkle_axis_angle)))

            # ####### RightShoulder
            smpl_rightShoulder_axis_angle = fullPose[0, 3*rightShoulder_inSMPL:3*rightShoulder_inSMPL+3].numpy() # 读rightShoulder的SMPL pose 参数  axis-angle表示
            # 解析出来的轴是世界坐标系下的，转角是通用的
            smpl_rightShoulder_world_axis, common_rightShoulder_angle = rvec2axisAndAngle(smpl_rightShoulder_axis_angle)
            # 线性变换矩阵，用于将世界坐标系下的轴转换到humanoid rightShoulder关节下的局部坐标系
            linear_transform_mat = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) 
            humanoid_rightShoulder_local_axis = linear_transform_mat @ smpl_rightShoulder_world_axis.reshape(3, 1)
            humanoid_rightShoulder_axis_angle = humanoid_rightShoulder_local_axis * common_rightShoulder_angle
            p.resetJointStateMultiDof(humanoid, rightShoulder, mat2pb(rvec2mat(humanoid_rightShoulder_axis_angle)))

            # ####### RightElbow
            smpl_rightElbow_axis_angle = fullPose[0, 3*rightElbow_inSMPL:3*rightElbow_inSMPL+3].numpy() # 读rightElbow的SMPL pose 参数  axis-angle表示
            # 解析出来的轴是世界坐标系下的，转角是通用的
            smpl_rightElbow_world_axis, common_rightElbow_angle = rvec2axisAndAngle(smpl_rightElbow_axis_angle)
            # 线性变换矩阵，用于将世界坐标系下的轴转换到humanoid rightElbow关节下的局部坐标系
            linear_transform_mat = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) 
            humanoid_rightElbow_local_axis = linear_transform_mat @ smpl_rightElbow_world_axis.reshape(3, 1)
            humanoid_rightElbow_axis_angle = humanoid_rightElbow_local_axis * common_rightElbow_angle
            p.resetJointStateMultiDof(humanoid, rightElbow, mat2pb(rvec2mat(humanoid_rightElbow_axis_angle)))

            # ####### RightWrist
            smpl_rightWrist_axis_angle = fullPose[0, 3*rightWrist_inSMPL:3*rightWrist_inSMPL+3].numpy()
            smpl_rightWrist_world_axis, common_rightWrist_angle = rvec2axisAndAngle(smpl_rightWrist_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 
            humanoid_rightWrist_local_axis = linear_transform_mat @ smpl_rightWrist_world_axis.reshape(3, 1)
            humanoid_rightWrist_axis_angle = humanoid_rightWrist_local_axis * common_rightWrist_angle
            p.resetJointStateMultiDof(humanoid, rightWrist, mat2pb(rvec2mat(humanoid_rightWrist_axis_angle)))

            # ####### LeftHip
            smpl_leftHip_axis_angle = fullPose[0, 3*leftHip_inSMPL:3*leftHip_inSMPL+3].numpy()
            smpl_leftHip_world_axis, common_leftHip_angle = rvec2axisAndAngle(smpl_leftHip_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            humanoid_leftHip_local_axis = linear_transform_mat @ smpl_leftHip_world_axis.reshape(3, 1)
            humanoid_leftHip_axis_angle = humanoid_leftHip_local_axis * common_leftHip_angle
            p.resetJointStateMultiDof(humanoid, leftHip, mat2pb(rvec2mat(humanoid_leftHip_axis_angle)))

            # ####### leftKnee
            smpl_leftKnee_axis_angle = fullPose[0, 3*leftKnee_inSMPL:3*leftKnee_inSMPL+3].numpy()
            smpl_leftKnee_world_axis, common_leftKnee_angle = rvec2axisAndAngle(smpl_leftKnee_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            humanoid_leftKnee_local_axis = linear_transform_mat @ smpl_leftKnee_world_axis.reshape(3, 1)
            humanoid_leftKnee_axis_angle = humanoid_leftKnee_local_axis * common_leftKnee_angle
            p.resetJointStateMultiDof(humanoid, leftKnee, mat2pb(rvec2mat(humanoid_leftKnee_axis_angle)))

            # ####### leftAnkle
            smpl_leftAnkle_axis_angle = fullPose[0, 3*leftAnkle_inSMPL:3*leftAnkle_inSMPL+3].numpy()
            smpl_leftAnkle_world_axis, common_leftAnkle_angle = rvec2axisAndAngle(smpl_leftAnkle_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            humanoid_leftAnkle_local_axis = linear_transform_mat @ smpl_leftAnkle_world_axis.reshape(3, 1)
            humanoid_leftAnkle_axis_angle = humanoid_leftAnkle_local_axis * common_leftAnkle_angle
            p.resetJointStateMultiDof(humanoid, leftAnkle, mat2pb(rvec2mat(humanoid_leftAnkle_axis_angle)))

            # ####### leftShoulder
            smpl_leftShoulder_axis_angle = fullPose[0, 3*leftShoulder_inSMPL:3*leftShoulder_inSMPL+3].numpy() # 读leftShoulder的SMPL pose 参数  axis-angle表示
            # 解析出来的轴是世界坐标系下的，转角是通用的
            smpl_leftShoulder_world_axis, common_leftShoulder_angle = rvec2axisAndAngle(smpl_leftShoulder_axis_angle)
            # 线性变换矩阵，用于将世界坐标系下的轴转换到humanoid leftShoulder关节下的局部坐标系
            linear_transform_mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) 
            humanoid_leftShoulder_local_axis = linear_transform_mat @ smpl_leftShoulder_world_axis.reshape(3, 1)
            humanoid_leftShoulder_axis_angle = humanoid_leftShoulder_local_axis * common_leftShoulder_angle
            p.resetJointStateMultiDof(humanoid, leftShoulder, mat2pb(rvec2mat(humanoid_leftShoulder_axis_angle)))

            # ####### leftElbow
            smpl_leftElbow_axis_angle = fullPose[0, 3*leftElbow_inSMPL:3*leftElbow_inSMPL+3].numpy() # 读leftElbow的SMPL pose 参数  axis-angle表示
            # 解析出来的轴是世界坐标系下的，转角是通用的
            smpl_leftElbow_world_axis, common_leftElbow_angle = rvec2axisAndAngle(smpl_leftElbow_axis_angle)
            # 线性变换矩阵，用于将世界坐标系下的轴转换到humanoid leftElbow关节下的局部坐标系
            linear_transform_mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) 
            humanoid_leftElbow_local_axis = linear_transform_mat @ smpl_leftElbow_world_axis.reshape(3, 1)
            humanoid_leftElbow_axis_angle = humanoid_leftElbow_local_axis * common_leftElbow_angle
            p.resetJointStateMultiDof(humanoid, leftElbow, mat2pb(rvec2mat(humanoid_leftElbow_axis_angle)))
            
            # ####### leftWrist
            smpl_leftWrist_axis_angle = fullPose[0, 3*leftWrist_inSMPL:3*leftWrist_inSMPL+3].numpy()
            smpl_leftWrist_world_axis, common_leftWrist_angle = rvec2axisAndAngle(smpl_leftWrist_axis_angle)
            linear_transform_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            humanoid_leftWrist_local_axis = linear_transform_mat @ smpl_leftWrist_world_axis.reshape(3, 1)
            humanoid_leftWrist_axis_angle = humanoid_leftWrist_local_axis * common_leftWrist_angle
            p.resetJointStateMultiDof(humanoid, leftWrist, mat2pb(rvec2mat(humanoid_leftWrist_axis_angle)))


if __name__ == '__main__':
    args = {
        'useYUp': True,

        'cameraDistance': 3,
        'cameraYaw': 180,
        'cameraPitch': -20,
        'cameraTargetPosition': [1, 1, 1],

        'data_folder': 'E:\\Human-Training-v3.2',

        'urdf': './data/urdf/humanoid_SMPL_thin.urdf',
        'smpl_mesh_path': './data/obj/T-pose smpl.obj',
        'save_mesh_root': './data/obj/',
        'smpl_model_root': './data/smpl/'
    }
    main(**args)