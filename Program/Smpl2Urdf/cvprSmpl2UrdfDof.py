import numpy as np
from torch import zeros
import smplx
import torch
import sys
sys.path.append('./')
import utils.urdf_utils as urdf_utils
import math
from scipy.spatial.transform import Rotation as R

class smpl2Urdf(object):
    def __init__(self):
        self.iner = 0.001
        self.ankle_size = [0.0875,0.06,0.185]
        self.lankle_offset = [0.01719,-0.06032,0.02617]
        self.rankle_offset = [-0.01719,-0.06032,0.02617]
        self.lowerback_offset = [0.0,0.05,0.013]
        self.upperback_offset = [0.0,0.02246,0.00143]
        self.chest_offset = np.array([0.0,0.057,-0.00687])
        self.chest_det = np.array([0.045,0,0])
        self.upperneck_length = 0.035
        self.dotmass = 0.0001
        self.limit = {
            'effort':1000.0,
            'lower':-3.14,
            'upper':3.14,
            'velocity':0.5
        }
        self.mass = [5,5,3,1,5,3,1,5,5,8,0.5,3,1,2,1,0.5,1,2,1,0.5] # 所有link
        self.isCal = [
            False,True,True,False,True,True,False,
            False,False,False,True,True,
            True,True,True,False,True,True,True,False,
        ] # 所有link
        self.weigh = [
            -1,0.05,0.05,-1,0.05,0.05,-1,
            0.065,0.05,0.07,0.03,0.06,
            0.04,0.05,0.05,0.04,0.04,0.05,0.05,0.04
        ] # 所有link
        self.shape = [
            -1, 'capsule', 'capsule', 'box', 'capsule', 'capsule', 'box',
            'sphere', 'sphere', 'sphere', 'capsule', 'capsule',
            'capsule', 'box', 'box', 'sphere', 'capsule', 'box', 'box', 'sphere',
        ]
        self.name = [
            'root', 'lhip', 'lknee', 'lankle', 'rhip', 'rknee', 'rankle',
            'lowerback', 'upperback', 'chest', 'lowerneck', 'upperneck',
            'lclavicle', 'lshoulder', 'lelbow', 'lwrist', 'rclavicle', 'rshoulder', 'relbow', 'rwrist',
        ]
        self.parentname = [
            -1, 'root', 'lhip', 'lknee', 'root', 'rhip', 'rknee', 
            'root', 'lowerback', 'upperback','chest', 'lowerneck', 
            'chest', 'lclavicle', 'lshoulder', 'lelbow', 'chest', 'rclavicle', 'rshoulder', 'relbow'
        ]
        self.parentIdx = [
            -1,0,0,0,1,2,3,4,5,6,7,8,
            9,9,9,12,13,14,16,17,18,19,20,21] # 24,smpl中每个joint的父joint
        self.childIdx = [
            -1,4,5,6,7,8,9,10,11,-1,-1,-1,
            15,16,17,-1,18,19,20,21,22,23,-1,-1] # 24,smpl中每个joint的子joint
        self.jointInUrdfIdx = [1,4,7,2,5,8,3,6,9,12,15,13,16,18,20,14,17,19,21] # 每个数代表smpl中的节点序号

    def CalRotFromVecs(self,vec1, vec2):
        '''
        计算从vec1旋转到vec2所需的旋转矩阵
        vec1:n
        vec2:n
        return:3*3 np.array
        '''
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
        rotaxis = np.cross(vec1,vec2)
        rotaxis = rotaxis / np.linalg.norm(rotaxis)
        sita = math.acos(np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        r = R.from_rotvec(rotaxis*sita)
        return r

    def __call__(self,savePath,shape=None):
        if shape is None:
            shape = np.zeros((10))
        model = smplx.create(
            R'./data/smplData/body_models',
            'smpl',
            gender='NEUTRAL')
        output = model(
            betas = torch.tensor(shape[None,:].astype(np.float32)),
            body_pose = torch.tensor(np.zeros((1,69)).astype(np.float32)),
            global_orient = torch.tensor(np.zeros((1,3)).astype(np.float32)),
            transl = torch.tensor(np.zeros((1,3)).astype(np.float32)))
        _,jointsPos,_=output.vertices.detach().cpu().numpy().squeeze(),output.joints.detach().cpu().numpy().squeeze(),model.faces + 1

        file = open(savePath, 'w')
        urdf_utils.write_start(file, 'amass')

        # base link
        base_link = urdf_utils.Link('base')
        base_link.iners.append(urdf_utils.Inertial(-jointsPos[0], [0,0,0], self.dotmass, self.iner))
        # base_link.iners.append(urdf_utils.Inertial([0,0,0], [0,0,0], self.dotmass, self.iner))
        base_link.writeFile(file)
        # root link
        root_link = urdf_utils.Link('root')
        root_link.iners.append(urdf_utils.Inertial([0,0,0], [0,0,0], self.mass[0], self.iner))
        root_link.colls.append(urdf_utils.Collision([0.00354, 0.065, -0.03107], [0, 1.5708, 0], 'collision_0_root', urdf_utils.Geometry('sphere', 0.05, 0.115)))
        root_link.colls.append(urdf_utils.Collision([-0.05769, -0.02577, -0.0174], [0, 0, 0], 'collision_1_root', urdf_utils.Geometry('sphere', 0.075)))
        root_link.colls.append(urdf_utils.Collision([0.06735, -0.02415, -0.0174], [0, 0, 0], 'collision_2_root', urdf_utils.Geometry('sphere', 0.075)))
        root_link.writeFile(file)
        # root joint
        joint = urdf_utils.Joint('root', 'floating', 'base', 'root', [0,0,0], [0,0,0], [0,1,0])
        joint.writeFile(file)

        for key, i in enumerate(self.jointInUrdfIdx):
            if i == 12: # neck
                jointpos = jointsPos[i]
                parentpos = jointsPos[self.parentIdx[i]]
                childpos = jointsPos[self.childIdx[i]]
                weigth = self.weigh[key+1]
                length = np.linalg.norm(jointpos-childpos)-2*weigth
                shape = self.shape[key+1]
                name = self.name[key+1]
                mass = self.mass[key+1]

                link = urdf_utils.Link(name+'_rx')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name+'_ry')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name+'_rz')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name)
                link.iners.append(urdf_utils.Inertial([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]],[0,0,0],mass,self.iner))
                rotvec = np.array([0,1,0])
                r = self.CalRotFromVecs(np.array([0,0,1]), rotvec)
                if shape == 'capsule':
                    geo = urdf_utils.Geometry('capsule', weigth, length)
                elif shape == 'box':
                    geo = urdf_utils.Geometry('box', weigth, weigth, length+1.6*weigth)
                link.colls.append(urdf_utils.Collision([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]], r.as_euler('xyz'), name, geo))
                link.writeFile(file)
                joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)

                continue
            if i == 15: # head
                jointpos = jointsPos[i]
                parentpos = jointsPos[self.parentIdx[i]]
                childpos = jointsPos[self.childIdx[i]]
                weigth = self.weigh[key+1]
                #length = np.linalg.norm(jointpos-childpos)-2*weigth
                length = self.upperneck_length
                shape = self.shape[key+1]
                name = self.name[key+1]
                mass = self.mass[key+1]

                link = urdf_utils.Link(name+'_rx')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name+'_ry')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name+'_rz')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name)
                link.iners.append(urdf_utils.Inertial([0, 0, 0],[0,0,0],mass,self.iner))
                rotvec = np.array([0,1,0])
                r = self.CalRotFromVecs(np.array([0,0,1]), rotvec)
                if shape == 'capsule':
                    geo = urdf_utils.Geometry('capsule', weigth, length)
                elif shape == 'box':
                    geo = urdf_utils.Geometry('box', weigth, weigth, length+1.6*weigth)
                link.colls.append(urdf_utils.Collision([0, 0, 0], r.as_euler('xyz'), name, geo))
                link.writeFile(file)
                joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)

                continue
            if self.isCal[key+1]:
                jointpos = jointsPos[i]
                parentpos = jointsPos[self.parentIdx[i]]
                childpos = jointsPos[self.childIdx[i]]
                weigth = self.weigh[key+1]
                length = np.linalg.norm(jointpos-childpos)-2*weigth
                shape = self.shape[key+1]
                name = self.name[key+1]
                mass = self.mass[key+1]

                link = urdf_utils.Link(name+'_rx')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name+'_ry')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name+'_rz')
                link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                link.writeFile(file)
                limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = urdf_utils.Link(name)
                link.iners.append(urdf_utils.Inertial((childpos-jointpos)/2.0,[0,0,0],mass,self.iner))
                rotvec = np.array(childpos-jointpos)
                r = self.CalRotFromVecs(np.array([0,0,1]), rotvec)
                if shape == 'capsule':
                    geo = urdf_utils.Geometry('capsule', weigth, length)
                elif shape == 'box':
                    if (i == 16) or (i == 17) or (i == 18) or (i == 19):
                        geo = urdf_utils.Geometry('box', length+1.6*weigth, weigth, weigth)
                    else:
                        geo = urdf_utils.Geometry('box', weigth, weigth, length+1.6*weigth)
                if (i == 16) or (i == 17) or (i == 18) or (i == 19): 
                    link.colls.append(urdf_utils.Collision((childpos-jointpos)/2.0, [0,0,0], name, geo))
                else:
                    link.colls.append(urdf_utils.Collision((childpos-jointpos)/2.0, r.as_euler('xyz'), name, geo))
                link.writeFile(file)
                joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)
            else:
                weigth = self.weigh[key+1]
                shape = self.shape[key+1]
                name = self.name[key+1]
                mass = self.mass[key+1]
                jointpos = jointsPos[i]
                parentpos = jointsPos[self.parentIdx[i]]
                if name ==  'lankle':
                    link = urdf_utils.Link(name+'_rx')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_ry')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_rz')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name)
                    link.iners.append(urdf_utils.Inertial(self.lankle_offset,[0,0,0],mass,self.iner))
                    geo = urdf_utils.Geometry('box', self.ankle_size[0], self.ankle_size[1], self.ankle_size[2])
                    link.colls.append(urdf_utils.Collision(self.lankle_offset,[0,0,0],name,geo))
                    link.writeFile(file)
                    joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                    joint.writeFile(file)
                elif name == 'rankle':
                    link = urdf_utils.Link(name+'_rx')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_ry')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_rz')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name)
                    link.iners.append(urdf_utils.Inertial(self.rankle_offset,[0,0,0],mass,self.iner))
                    geo = urdf_utils.Geometry('box', self.ankle_size[0], self.ankle_size[1], self.ankle_size[2])
                    link.colls.append(urdf_utils.Collision(self.rankle_offset,[0,0,0],name,geo))
                    link.writeFile(file)
                    joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                    joint.writeFile(file)
                elif name == 'lowerback':
                    link = urdf_utils.Link(name+'_rx')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_ry')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_rz')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name)
                    link.iners.append(urdf_utils.Inertial(self.lowerback_offset,[0,0,0],mass,self.iner))
                    geo = urdf_utils.Geometry(shape,weigth)
                    link.colls.append(urdf_utils.Collision(self.lowerback_offset,[0,0,0],name,geo))
                    link.writeFile(file)
                    joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                    joint.writeFile(file)
                elif name == 'upperback':
                    link = urdf_utils.Link(name+'_rx')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_ry')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_rz')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name)
                    link.iners.append(urdf_utils.Inertial(self.upperback_offset,[0,0,0],mass,self.iner))
                    geo = urdf_utils.Geometry(shape,weigth)
                    link.colls.append(urdf_utils.Collision(self.upperback_offset,[0,0,0],name,geo))
                    link.writeFile(file)
                    joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                    joint.writeFile(file)
                elif name == 'chest':
                    link = urdf_utils.Link(name+'_rx')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rx', 'revolute', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_ry')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_rz')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name)
                    link.iners.append(urdf_utils.Inertial(self.chest_offset,[0,0,0],mass,self.iner))
                    geo = urdf_utils.Geometry(shape,weigth)
                    link.colls.append(urdf_utils.Collision(self.chest_offset+self.chest_det,[0,0,0],name+'0',geo))
                    link.colls.append(urdf_utils.Collision(self.chest_offset-self.chest_det,[0,0,0],name+'1',geo))
                    link.writeFile(file)
                    joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                    joint.writeFile(file)
                elif name == 'lwrist':
                    link = urdf_utils.Link(name+'_rx')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rx', 'fixed', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_ry')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_ry', 'fixed', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_rz')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rz', 'fixed', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name)
                    link.iners.append(urdf_utils.Inertial([weigth,0,0],[0,0,0],mass,self.iner))
                    geo = urdf_utils.Geometry(shape,weigth)
                    link.colls.append(urdf_utils.Collision([weigth,0,0],[0,0,0],name,geo))
                    link.writeFile(file)
                    joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                    joint.writeFile(file)
                elif name == 'rwrist':
                    link = urdf_utils.Link(name+'_rx')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rx', 'fixed', self.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_ry')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_ry', 'fixed', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name+'_rz')
                    link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],self.dotmass,self.iner))
                    link.writeFile(file)
                    limit = urdf_utils.Limit(self.limit['effort'], self.limit['lower'], self.limit['upper'], self.limit['velocity'])
                    joint = urdf_utils.Joint(name+'_rz', 'fixed', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                    joint.writeFile(file)

                    link = urdf_utils.Link(name)
                    link.iners.append(urdf_utils.Inertial([-1*weigth,0,0],[0,0,0],mass,self.iner))
                    geo = urdf_utils.Geometry(shape,weigth)
                    link.colls.append(urdf_utils.Collision([-1*weigth,0,0],[0,0,0],name,geo))
                    link.writeFile(file)
                    joint = urdf_utils.Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                    joint.writeFile(file)

        urdf_utils.write_end(file)
        file.close()

if __name__ == '__main__':

    import pickle as pkl
    pklPath = R'C:\Users\yangyuan\Desktop\00000.pkl'
    with open(pklPath, 'rb') as file:
        data = pkl.load(file)
    betas = data['person00']['betas']

    test = smpl2Urdf()
    test(
        'test.urdf',
        betas)
    pass