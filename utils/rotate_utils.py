from scipy.spatial.transform import Rotation as R
import numpy as np
import math

def Camera_project(points, externalMat=None, internalMat=None):
    '''
    points: n * 3 np.array
    return: ex_: n * 3
            in_: n * 2
    '''
    if externalMat is None:
        return points
    else:
        pointsInCamera = np.dot(externalMat, np.row_stack((points.T, np.ones(points.__len__()))))
        if internalMat is None:
            return pointsInCamera[:3,:].T
        else:
            pointsInImage = np.dot(internalMat, pointsInCamera[:3,:]) / pointsInCamera[2,:][None,:]
            return pointsInImage[:2,:].T

def CalRotFromVecs(vec1, vec2):
    '''
    计算从vec1旋转到vec2所需的旋转矩阵
    '''
    rotaxis = np.cross(vec1,vec2)
    rotaxis = rotaxis / np.linalg.norm(rotaxis)
    sita = math.acos(np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    r = R.from_rotvec(rotaxis*sita)
    #rotvec = r.apply(vec1)
    return r

def planeFit(points):
    '''
    从points中拟合一个平面方程
    z = a[0]*x + a[1]*y + a[2]
    '''
    A11 = 0
    A12 = 0
    A13 = 0
    A21 = 0
    A22 = 0
    A23 = 0
    A31 = 0
    A32 = 0
    A33 = 0
    S0 = 0
    S1 = 0
    S2 = 0
    for i in range(points.shape[0]):
        A11 += (points[i][0]*points[i][0])
        A12 += (points[i][0]*points[i][1])
        A13 += (points[i][0])
        A21 += (points[i][0]*points[i][1])
        A22 += (points[i][1]*points[i][1])
        A23 += (points[i][1])
        A31 += (points[i][0])
        A32 += (points[i][1])
        A33 += (1)
        S0 += (points[i][0]*points[i][2])
        S1 += (points[i][1]*points[i][2])
        S2 += (points[i][2])

    a = np.linalg.solve([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]],[S0,S1,S2])
    return a

def readVclCamparams(path):
    camIns = []
    camExs = []
    with open(path, 'r') as f:
        line = f.readline()
        camIdx = 0
        while(line):
            if line == (str(camIdx)+'\n'):
                camIdx += 1
                camIn = []
                camEx = []
                for i in range(3):
                    line = f.readline().strip().split()
                    camIn.append([float(line[0]), float(line[1]), float(line[2])])
                camIns.append(camIn)
                _ = f.readline()
                for i in range(3):
                    line = f.readline().strip().split()
                    camEx.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
                camEx.append([0.0, 0.0, 0.0, 1.0])
                camExs.append(camEx)
            line = f.readline()
    return camIns, camExs

def writeVclCamparams(path, camIns, camExs):
    with open(path, 'w') as f:
        camIdx = 0
        for camIn, camEx in zip(camIns, camExs):
            f.write(str(camIdx)+'\n')
            camIdx += 1
            for i in range(3):
                f.write(str(camIn[i][0]) + ' ' + str(camIn[i][1]) + ' ' + str(camIn[i][2]) + '\n')
            f.write('0 0\n')
            for i in range(3):
                f.write(str(camEx[i][0]) + ' ' + str(camEx[i][1]) + ' ' + str(camEx[i][2]) + ' ' + str(camEx[i][3]) + '\n')

if __name__ == '__main__':
    import glob
    import os
    import json
    path = r'E:\Evaluations_CVPR2022\Eval_Human36M10FPS'
    Human36 = ['S9', 'S11']
    for id in Human36:
        squenceIds = glob.glob(os.path.join(path,id,'camparams','*'))
        for squenceId in squenceIds:
            camPath = os.path.join(squenceId, 'camparams.txt')
            camIns, camExs = readVclCamparams(camPath)
            with open(os.path.join(squenceId, 'camIn0.json'), 'w') as f:
                f.write(json.dumps(camIns[0]))
            print(1)