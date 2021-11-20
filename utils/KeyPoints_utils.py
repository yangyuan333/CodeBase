class Openpose25(object):
    Idx2Name = {
        0:'Nose',
        1:'Neck',
        2:'RShoulder',
        3:'RElbow',
        4:'RWrist',
        5:'LShoulder',
        6:'LElbow',
        7:'LWrist',
        8:'Hip',
        9:'RHip',
        10:'RKnee',
        11:'RAnkle',
        12:'LHip',
        13:'LKnee',
        14:'LAnkle',
        15:'REye',
        16:'LEye',
        17:'REar',
        18:'LEar',
        19:'LBigToe',
        20:'LSmallToe',
        21:'LHeel',
        22:'RBigToe',
        23:'RSmallToe',
        24:'RHeel'
    }
    Name2Idx = {
        'Nose':0,
        'Neck':1,
        'RShoulder':2,
        'RElbow':3,
        'RWrist':4,
        'LShoulder':5,
        'LElbow':6,
        'LWrist':7,
        'MidHip':8,
        'RHip':9,
        'RKnee':10,
        'RAnkle':11,
        'LHip':12,
        'LKnee':13,
        'LAnkle':14,
        'REye':15,
        'LEye':16,
        'REar':17,
        'LEar':18,
        'LBigToe':19,
        'LSmallToe':20,
        'LHeel':21,
        'RBigToe':22,
        'RSmallToe':23,
        'RHeel':24
    }

class Halpe(object):
    Idx2Name = {
        0:'Nose',
        1:'LEye',
        2:'REye',
        3:'LEar',
        4:'REar',
        5:'LShoulder',
        6:'RShoulder',
        7:'LElbow',
        8:'RElbow',
        9:'LWrist',
        10:'RWrist',
        11:'LHip',
        12:'RHip',
        13:'LKnee',
        14:'RKnee',
        15:'LAnkle',
        16:'RAnkle',
        17:'Head',
        18:'Neck',
        19:'Hip',
        20:'LBigToe',
        21:'RBigToe',
        22:'LSmallToe',
        23:'RSmallToe',
        24:'LHeel',
        25:'RHeel'
    }
    Name2Idx = {
        'Nose':0,
        'LEye':1,
        'REye':2,
        'LEar':3,
        'REar':4,
        'LShoulder':5,
        'RShoulder':6,
        'LElbow':7,
        'RElbow':8,
        'LWrist':9,
        'RWrist':10,
        'LHip':11,
        'RHip':12,
        'LKnee':13,
        'RKnee':14,
        'LAnkle':15,
        'RAnkle':16,
        'Head':17,
        'Neck':18,
        'Hip':19,
        'LBigToe':20,
        'RBigToe':21,
        'LSmallToe':22,
        'RSmallToe':23,
        'LHeel':24,
        'RHeel':25
    }

def Halpe2Openpose25(jointsInHalpe):
    JointsInOpenpose25 = []
    for name in Openpose25.Idx2Name.values():
        JointsInOpenpose25.append(jointsInHalpe[Halpe.Name2Idx[name]])
    return JointsInOpenpose25

if __name__ == '__main__':
    import json
    import numpy as np
    import glob
    import os

    path = r'E:\Evaluations_CVPR2022\Eval_GPA'
    squenceIds = glob.glob(os.path.join(path, 'keypoints', '*'))
    for squenceId in squenceIds:
        cameraIds = glob.glob(os.path.join(squenceId, '*'))
        for cameraId in cameraIds:
            savePath = os.path.join(path, 'keypoints25', os.path.basename(squenceId), os.path.basename(cameraId))
            os.makedirs(savePath, exist_ok=True)
            frameIds = glob.glob(os.path.join(cameraId, '*'))
            for frameId in frameIds:
                with open(frameId, 'r') as f:
                    HumanKeypoint = json.load(f)
                HumanJoints = np.array(HumanKeypoint['people'][0]['pose_keypoints_2d']).reshape(-1,3)
                OpenposeJoints = Halpe2Openpose25(HumanJoints)
                HumanKeypoint['people'][0]['pose_keypoints_2d'] = list(np.array(OpenposeJoints).reshape(-1))
                with open(os.path.join(savePath, os.path.basename(frameId)), 'w') as f:
                    json.dump(HumanKeypoint, f)

    # with open(r'E:\Evaluations_CVPR2022\Eval_Human36M10FPS\S9\keypoints\Directions\Camera00\00416_keypoints.json', 'r') as f:
    #     HumanKeypoint = json.load(f)
    # HumanJoints = np.array(HumanKeypoint['people'][0]['pose_keypoints_2d']).reshape(-1,3)
    # OpenposeJoints = Halpe2Openpose25(HumanJoints)
    # import cv2
    # img = cv2.imread(r'E:\Evaluations_CVPR2022\Eval_Human36M10FPS\S9\images\Directions\Camera00\00416.jpg')
    # for joint in OpenposeJoints:
    #     img = cv2.circle(img, (int(joint[0]), int(joint[1])), 5, (0,0,255), -1)
    #     cv2.imshow('1', img)
    #     cv2.waitKey(0)