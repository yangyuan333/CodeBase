import glob
import os
import sys
sys.path.append('./')
import pickle as pkl
import shutil
import numpy as np
import torch
from utils.smpl_utils import SMPLModel,pkl2Smpl
from utils.rotate_utils import *


class Link(object):
    def __init__(self, name=None):
        self.iners = []
        self.colls = []
        if name is not None:
            self.name = name
    def writeFile(self, file):
        file.write("        <link name=\""+ self.name +"\">\n")

        for iner in self.iners:
            iner.writeFile(file)
        for coll in self.colls:
            coll.writeFile(file)

        file.write("        </link>\n")

class Inertial(object):
    def __init__(self, xyz=None, rpy=None, mass=None, inertia=None):
        if xyz is not None:
            self.xyz = xyz
        if rpy is not None:
            self.rpy = rpy
        if mass is not None:
            self.mass = mass
        if inertia is not None:
            self.inertia = inertia       
    def writeFile(self, file):
        file.write("            <inertial>\n")
        file.write("                <origin xyz=\""+ str(self.xyz[0]) + ' ' + str(self.xyz[1]) + ' ' + str(self.xyz[2]) + "\" rpy=\""+ str(self.rpy[0]) + ' ' + str(self.rpy[1]) + ' ' + str(self.rpy[2]) +"\"/>\n")
        file.write("                <mass value=\"" + str(self.mass) + "\"/>\n")
        file.write("                <inertia ixx=\""+ str(self.inertia) +"\" ixy=\"0\" ixz=\"0\" iyy=\""+ str(self.inertia) +"\" iyz=\"0\" izz=\""+ str(self.inertia) +"\"/>\n")
        file.write("            </inertial>\n")

class Geometry(object):
    def __init__(self, name=None, a=None, b=None, c=None):
        if name is not None:
            self.name = name
        self.geo = []
        if a is not None:
            self.geo.append(a)
        if b is not None:
            self.geo.append(b)
        if c is not None:
            self.geo.append(c)
    def writeFile(self, file):
        if self.name == 'sphere':
            if self.geo.__len__() == 1:
                file.write("                    <sphere radius=\""+ str(self.geo[0]) +"\"/>\n")
            else:
                file.write("                    <sphere radius=\""+ str(self.geo[0]) +"\" length=\"" + str(self.geo[1]) + "\"/>\n")
        elif self.name == 'capsule':
            file.write("                    <capsule radius=\""+ str(self.geo[0]) +"\" length=\"" + str(self.geo[1]) + "\"/>\n")
        elif self.name == 'box':
            file.write("                    <box size=\""+ str(self.geo[0]) + str(' ') + str(self.geo[1]) + str(' ') + str(self.geo[2]) + "\"/>\n")
        elif self.name == 'cylinder':
            file.write("                    <cylinder radius=\""+ str(self.geo[0]) +"\" length=\"" + str(self.geo[1]) + "\"/>\n")

class Collision(object):
    def __init__(self, xyz=None, rpy=None, name=None, geometry=None):
        if xyz is not None:
            self.xyz = xyz
        if rpy is not None:
            self.rpy = rpy
        if name is not None:
            self.name = name
        if geometry is not None:
            self.geometry = geometry           
    def writeFile(self,file):
        file.write("            <collision name=\"" + self.name + "\">\n")
        file.write("                <origin xyz=\""+ str(self.xyz[0]) + ' ' + str(self.xyz[1]) + ' ' + str(self.xyz[2]) + "\" rpy=\""+ str(self.rpy[0]) + ' ' + str(self.rpy[1]) + ' ' + str(self.rpy[2]) +"\"/>\n")
        file.write("                <geometry>\n")

        self.geometry.writeFile(file)

        file.write("                </geometry>\n")
        file.write("            </collision>\n") 

class Limit(object):
    def __init__(self, effort=None, lower=None, upper=None, velocity=None):
        self.effort = effort
        self.lower = lower
        self.upper = upper
        self.velocity = velocity
    def writeFile(self, file):
        file.write("            <limit effort=\""+ str(self.effort) +"\" lower=\"" + str(self.lower) +"\" upper=\"" + str(self.upper) + "\" velocity=\"" + str(self.velocity) + "\" />\n")

class Joint(object):
    def __init__(self, name=None, type=None, parent=None, child=None, xyz=None, rpy=None, axis=None, limit=None):
        if name is not None:
            self.name = name
        if type is not None:
            self.type = type
        if parent is not None:
            self.parent = parent
        if child is not None:
            self.child = child
        if xyz is not None:
            self.xyz = xyz
        if rpy is not None:
            self.rpy = rpy
        if axis is not None:
            self.axis = axis
        if limit is not None:
            self.limit = limit
    def writeFile(self, file):
        file.write("        <joint name=\""+ self.name + "\" type=\"" + self.type +"\">\n")
        file.write("            <origin xyz=\""+ str(self.xyz[0]) + ' ' + str(self.xyz[1]) + ' ' + str(self.xyz[2]) + "\" rpy=\""+ str(self.rpy[0]) + ' ' + str(self.rpy[1]) + ' ' + str(self.rpy[2]) +"\"/>\n")
        file.write("            <parent link=\""+ self.parent +"\"/>\n")
        file.write("            <child link=\""+ self.child +"\"/>\n")
        if self.type == 'revolute':
            self.limit.writeFile(file)
        if self.type == 'fixed':
            pass
        elif self.type == 'floating':
            pass
        else:
            file.write("            <axis xyz=\""+ str(self.axis[0]) + ' ' + str(self.axis[1]) + ' ' + str(self.axis[2]) +"\"/>\n")
        file.write("        </joint>\n")

def write_start(file,robotName):
    file.write('<?xml version="1.0"?>\n')
    file.write('    <robot name="'+ robotName +'">\n')
def write_end(file):
    file.write('    </robot>')
def write_rootLink(file):
    file.write("        <link name=\"root\">\n")
    file.write("            <inertial>\n")
    file.write("                <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n")
    file.write("                <mass value=\"5.0\"/>\n")
    file.write("                <inertia ixx=\"0.001\" ixy=\"0\" ixz=\"0\" iyy=\"0.001\" iyz=\"0\" izz=\"0.001\"/>\n")
    file.write("            </inertial>\n")
    file.write("            <collision name=\"collision_0_root\">\n")
    file.write("                <origin xyz=\"0.00354 0.065 -0.03107\" rpy=\"0 1.5708 0\"/>\n")
    file.write("                <geometry>\n")
    file.write("                    <sphere radius=\"0.05\" length=\"0.115\"/>\n")
    file.write("                </geometry>\n")
    file.write("            </collision>\n")
    file.write("            <collision name=\"collision_1_root\">\n")
    file.write("                <origin xyz=\"-0.05769 -0.02577 -0.0174\" rpy=\"0 0 0\"/>\n")
    file.write("                <geometry>\n")
    file.write("                    <sphere radius=\"0.075\"/>\n")
    file.write("                </geometry>\n")
    file.write("            </collision>\n")
    file.write("            <collision name=\"collision_2_root\">\n")
    file.write("                <origin xyz=\"0.06735 -0.02415 -0.0174\" rpy=\"0 0 0\"/>\n")
    file.write("                <geometry>\n")
    file.write("                    <sphere radius=\"0.075\"/>\n")
    file.write("                </geometry>\n")
    file.write("            </collision>\n")
    file.write("        </link>\n")
def write_joint():
    pass
def write_link():
    pass

class Config(object):
    # urdfPath = r'./data/temdata/results/demo3.urdf'
    # transPath = r'./data/temdata/results/demo3.txt'
    iner = 0.001
    ankle_size = [0.0875,0.06,0.185]
    lankle_offset = [0.01719,-0.06032,0.02617]
    rankle_offset = [-0.01719,-0.06032,0.02617]
    lowerback_offset = [0.0,0.05,0.013]
    upperback_offset = [0.0,0.02246,0.00143]
    chest_offset = np.array([0.0,0.057,-0.00687])
    chest_det = np.array([0.045,0,0])
    upperneck_length = 0.035
    mass = [5,5,3,1,5,3,1,5,5,8,0.5,3,1,2,1,0.5,1,2,1,0.5] # 所有link
    dotmass = 0.0001
    limit = {
        'effort':1000.0,
        'lower':-3.14,
        'upper':3.14,
        'velocity':0.5
    }
    isCal = [
        False,True,True,False,True,True,False,
        False,False,False,True,True,
        True,True,True,False,True,True,True,False,
    ] # 所有link
    weigh = [
        -1,0.05,0.05,-1,0.05,0.05,-1,
        0.065,0.05,0.07,0.03,0.06,
        0.04,0.05,0.05,0.04,0.04,0.05,0.05,0.04
    ] # 所有link
    shape = [
        -1, 'capsule', 'capsule', 'box', 'capsule', 'capsule', 'box',
        'sphere', 'sphere', 'sphere', 'capsule', 'capsule',
        'capsule', 'box', 'box', 'sphere', 'capsule', 'box', 'box', 'sphere',
    ]
    name = [
        'root', 'lhip', 'lknee', 'lankle', 'rhip', 'rknee', 'rankle',
        'lowerback', 'upperback', 'chest', 'lowerneck', 'upperneck',
        'lclavicle', 'lshoulder', 'lelbow', 'lwrist', 'rclavicle', 'rshoulder', 'relbow', 'rwrist',
    ]
    parentname = [
        -1, 'root', 'lhip', 'lknee', 'root', 'rhip', 'rknee', 
        'root', 'lowerback', 'upperback','chest', 'lowerneck', 
        'chest', 'lclavicle', 'lshoulder', 'lelbow', 'chest', 'rclavicle', 'rshoulder', 'relbow'
    ]

def smpl2Urdf(config):
    smplModel = SMPLModel()

    with open(config['pklPath'], 'rb') as file:
        data = pkl.load(file)
    if 'person00' in data:
        beta = data['person00']['betas']
    else:
        beta = data['betas']
    if beta.ndim == 1:
        beta = beta[None,:].astype(np.float32)
    else:
        beta = beta.astype(np.float32)
    smplModel = SMPLModel()
    smpl_vs, smpl_js = smplModel(
        torch.tensor(beta),
        torch.tensor(np.zeros((1, 72)).astype(np.float32)), 
        torch.tensor(np.zeros((1, 3)).astype(np.float32)),
        torch.tensor([[1.0]])
    )
    smpl_vs, smpl_js = smpl_vs[0].numpy(), smpl_js[0].numpy()

    urdfPath = config['urdfPath']

    parentIdx = [
        -1,0,0,0,1,2,3,4,5,6,7,8,
        9,9,9,12,13,14,16,17,18,19,20,21] # 24,smpl中每个joint的父joint
    childIdx = [
        -1,4,5,6,7,8,9,10,11,-1,-1,-1,
        15,16,17,-1,18,19,20,21,22,23,-1,-1] # 24,smpl中每个joint的子joint
    jointsPos = smpl_js # 24*3
    jointInUrdfIdx = [1,4,7,2,5,8,3,6,9,12,15,13,16,18,20,14,17,19,21] # 每个数代表smpl中的节点序号

    file = open(urdfPath, 'w')
    write_start(file, 'amass')

    #root link
    root_link = Link('root')
    np.savetxt(config['transPath'], -jointsPos[0])
    if config['isZero']:
        root_link.iners.append(Inertial([0,0,0], [0,0,0], Config.mass[0], Config.iner))
    else:
        root_link.iners.append(Inertial(-jointsPos[0], [0,0,0], Config.mass[0], Config.iner))
    root_link.colls.append(Collision([0.00354, 0.065, -0.03107], [0, 1.5708, 0], 'collision_0_root', Geometry('sphere', 0.05, 0.115)))
    root_link.colls.append(Collision([-0.05769, -0.02577, -0.0174], [0, 0, 0], 'collision_1_root', Geometry('sphere', 0.075)))
    root_link.colls.append(Collision([0.06735, -0.02415, -0.0174], [0, 0, 0], 'collision_2_root', Geometry('sphere', 0.075)))
    root_link.writeFile(file)

    for key, i in enumerate(jointInUrdfIdx):
        if i == 12:
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            childpos = jointsPos[childIdx[i]]
            weigth = Config.weigh[key+1]
            length = np.linalg.norm(jointpos-childpos)-2*weigth
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]

            link = Link(name)
            link.iners.append(Inertial([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]],[0,0,0],mass,Config.iner))
            # link.iners.append(Inertial([0, 0, parentpos[2]-jointpos[2]],[0,0,0],mass,Config.iner))

            rotvec = np.array([0,1,0])
            r = CalRotFromVecs(np.array([0,0,1]), rotvec)
            if shape == 'capsule':
                geo = Geometry('capsule', weigth, length)
            elif shape == 'box':
                geo = Geometry('box', weigth, weigth, length+1.6*weigth)
            link.colls.append(Collision([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]], r.as_euler('xyz'), name, geo))
            # link.colls.append(Collision([0, 0, parentpos[2]-jointpos[2]], r.as_euler('xyz'), name, geo))

            link.writeFile(file)

            joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])

            joint.writeFile(file)

            continue
        if i == 15:
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            childpos = jointsPos[childIdx[i]]
            weigth = Config.weigh[key+1]
            #length = np.linalg.norm(jointpos-childpos)-2*weigth
            length = Config.upperneck_length
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]

            link = Link(name)
            # link.iners.append(Inertial([0, (length+2*weigth)/2, 0],[0,0,0],mass,Config.iner))
            link.iners.append(Inertial([0, 0, 0],[0,0,0],mass,Config.iner))
            rotvec = np.array([0,1,0])
            r = CalRotFromVecs(np.array([0,0,1]), rotvec)
            if shape == 'capsule':
                geo = Geometry('capsule', weigth, length)
            elif shape == 'box':
                geo = Geometry('box', weigth, weigth, length+1.6*weigth)
            # link.colls.append(Collision([0, (length+2*weigth)/2, 0], r.as_euler('xyz'), name, geo))
            link.colls.append(Collision([0, 0, 0], r.as_euler('xyz'), name, geo))
            link.writeFile(file)

            joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])

            joint.writeFile(file)

            continue
        if Config.isCal[key+1]:
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            childpos = jointsPos[childIdx[i]]
            weigth = Config.weigh[key+1]
            length = np.linalg.norm(jointpos-childpos)-2*weigth
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]

            link = Link(name)
            link.iners.append(Inertial((childpos-jointpos)/2.0,[0,0,0],mass,Config.iner))

            rotvec = np.array(childpos-jointpos)
            r = CalRotFromVecs(np.array([0,0,1]), rotvec)
            if shape == 'capsule':
                geo = Geometry('capsule', weigth, length)
            elif shape == 'box':
                if (i == 16) or (i == 17) or (i == 18) or (i == 19):
                    geo = Geometry('box', length+1.6*weigth, weigth, weigth)
                else:
                    geo = Geometry('box', weigth, weigth, length+1.6*weigth)
            if (i == 16) or (i == 17) or (i == 18) or (i == 19): 
                link.colls.append(Collision((childpos-jointpos)/2.0, [0,0,0], name, geo))
            else:
                link.colls.append(Collision((childpos-jointpos)/2.0, r.as_euler('xyz'), name, geo))
            link.writeFile(file)

            joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])

            joint.writeFile(file)
        else:
            weigth = Config.weigh[key+1]
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            if name ==  'lankle':
                link = Link(name)
                link.iners.append(Inertial(Config.lankle_offset,[0,0,0],mass,Config.iner))
                geo = Geometry('box', Config.ankle_size[0], Config.ankle_size[1], Config.ankle_size[2])
                link.colls.append(Collision(Config.lankle_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'rankle':
                link = Link(name)
                link.iners.append(Inertial(Config.rankle_offset,[0,0,0],mass,Config.iner))
                geo = Geometry('box', Config.ankle_size[0], Config.ankle_size[1], Config.ankle_size[2])
                link.colls.append(Collision(Config.rankle_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'lowerback':
                link = Link(name)
                link.iners.append(Inertial(Config.lowerback_offset,[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision(Config.lowerback_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'upperback':
                link = Link(name)
                link.iners.append(Inertial(Config.upperback_offset,[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision(Config.upperback_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'chest':
                link = Link(name)
                link.iners.append(Inertial(Config.chest_offset,[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision(Config.chest_offset+Config.chest_det,[0,0,0],name+'0',geo))
                link.colls.append(Collision(Config.chest_offset-Config.chest_det,[0,0,0],name+'1',geo))
                link.writeFile(file)
                joint = Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'lwrist':
                link = Link(name)
                link.iners.append(Inertial([weigth,0,0],[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision([weigth,0,0],[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'rwrist':
                link = Link(name)
                link.iners.append(Inertial([-1*weigth,0,0],[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision([-1*weigth,0,0],[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
                joint.writeFile(file)

    write_end(file)
    file.close()

def smpl2UrdfDof(config):
    smplModel = SMPLModel()

    with open(config['pklPath'], 'rb') as file:
        data = pkl.load(file)
    if 'person00' in data:
        beta = data['person00']['betas']
    else:
        beta = data['betas']
    if beta.ndim == 1:
        beta = beta[None,:].astype(np.float32)
    else:
        beta = beta.astype(np.float32)
    smplModel = SMPLModel()
    smpl_vs, smpl_js = smplModel(
        torch.tensor(beta),
        torch.tensor(np.zeros((1, 72)).astype(np.float32)), 
        torch.tensor(np.zeros((1, 3)).astype(np.float32)),
        torch.tensor([[1.0]])
    )
    smpl_vs, smpl_js = smpl_vs[0].numpy(), smpl_js[0].numpy()

    urdfPath = config['urdfPath']

    parentIdx = [
        -1,0,0,0,1,2,3,4,5,6,7,8,
        9,9,9,12,13,14,16,17,18,19,20,21] # 24,smpl中每个joint的父joint
    childIdx = [
        -1,4,5,6,7,8,9,10,11,-1,-1,-1,
        15,16,17,-1,18,19,20,21,22,23,-1,-1] # 24,smpl中每个joint的子joint
    jointsPos = smpl_js # 24*3
    jointInUrdfIdx = [1,4,7,2,5,8,3,6,9,12,15,13,16,18,20,14,17,19,21] # 每个数代表smpl中的节点序号

    file = open(urdfPath, 'w')
    write_start(file, 'amass')

    # base link
    base_link = Link('base')
    if config['isZero']:
        base_link.iners.append(Inertial([0,0,0], [0,0,0], Config.dotmass, Config.iner))
    else:
       base_link.iners.append(Inertial(-jointsPos[0], [0,0,0], Config.dotmass, Config.iner)) 
    
    base_link.writeFile(file)
    # root link
    root_link = Link('root')
    root_link.iners.append(Inertial([0,0,0], [0,0,0], Config.mass[0], Config.iner))
    root_link.colls.append(Collision([0.00354, 0.065, -0.03107], [0, 1.5708, 0], 'collision_0_root', Geometry('sphere', 0.05, 0.115)))
    root_link.colls.append(Collision([-0.05769, -0.02577, -0.0174], [0, 0, 0], 'collision_1_root', Geometry('sphere', 0.075)))
    root_link.colls.append(Collision([0.06735, -0.02415, -0.0174], [0, 0, 0], 'collision_2_root', Geometry('sphere', 0.075)))
    root_link.writeFile(file)
    # root joint
    joint = Joint('root', 'floating', 'base', 'root', [0,0,0], [0,0,0], [0,1,0])
    joint.writeFile(file)

    for key, i in enumerate(jointInUrdfIdx):
        if i == 12: # neck
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            childpos = jointsPos[childIdx[i]]
            weigth = Config.weigh[key+1]
            length = np.linalg.norm(jointpos-childpos)-2*weigth
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]

            link = Link(name+'_rx')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
            joint.writeFile(file)

            link = Link(name+'_ry')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
            joint.writeFile(file)

            link = Link(name+'_rz')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
            joint.writeFile(file)

            link = Link(name)
            link.iners.append(Inertial([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]],[0,0,0],mass,Config.iner))
            rotvec = np.array([0,1,0])
            r = CalRotFromVecs(np.array([0,0,1]), rotvec)
            if shape == 'capsule':
                geo = Geometry('capsule', weigth, length)
            elif shape == 'box':
                geo = Geometry('box', weigth, weigth, length+1.6*weigth)
            link.colls.append(Collision([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]], r.as_euler('xyz'), name, geo))
            link.writeFile(file)
            joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
            joint.writeFile(file)

            continue
        if i == 15: # head
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            childpos = jointsPos[childIdx[i]]
            weigth = Config.weigh[key+1]
            #length = np.linalg.norm(jointpos-childpos)-2*weigth
            length = Config.upperneck_length
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]

            link = Link(name+'_rx')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
            joint.writeFile(file)

            link = Link(name+'_ry')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
            joint.writeFile(file)

            link = Link(name+'_rz')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
            joint.writeFile(file)

            link = Link(name)
            link.iners.append(Inertial([0, 0, 0],[0,0,0],mass,Config.iner))
            rotvec = np.array([0,1,0])
            r = CalRotFromVecs(np.array([0,0,1]), rotvec)
            if shape == 'capsule':
                geo = Geometry('capsule', weigth, length)
            elif shape == 'box':
                geo = Geometry('box', weigth, weigth, length+1.6*weigth)
            link.colls.append(Collision([0, 0, 0], r.as_euler('xyz'), name, geo))
            link.writeFile(file)
            joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
            joint.writeFile(file)

            continue
        if Config.isCal[key+1]:
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            childpos = jointsPos[childIdx[i]]
            weigth = Config.weigh[key+1]
            length = np.linalg.norm(jointpos-childpos)-2*weigth
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]

            link = Link(name+'_rx')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
            joint.writeFile(file)

            link = Link(name+'_ry')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
            joint.writeFile(file)

            link = Link(name+'_rz')
            link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
            link.writeFile(file)
            limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
            joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
            joint.writeFile(file)

            link = Link(name)
            link.iners.append(Inertial((childpos-jointpos)/2.0,[0,0,0],mass,Config.iner))
            rotvec = np.array(childpos-jointpos)
            r = CalRotFromVecs(np.array([0,0,1]), rotvec)
            if shape == 'capsule':
                geo = Geometry('capsule', weigth, length)
            elif shape == 'box':
                if (i == 16) or (i == 17) or (i == 18) or (i == 19):
                    geo = Geometry('box', length+1.6*weigth, weigth, weigth)
                else:
                    geo = Geometry('box', weigth, weigth, length+1.6*weigth)
            if (i == 16) or (i == 17) or (i == 18) or (i == 19): 
                link.colls.append(Collision((childpos-jointpos)/2.0, [0,0,0], name, geo))
            else:
                link.colls.append(Collision((childpos-jointpos)/2.0, r.as_euler('xyz'), name, geo))
            link.writeFile(file)
            joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
            joint.writeFile(file)
        else:
            weigth = Config.weigh[key+1]
            shape = Config.shape[key+1]
            name = Config.name[key+1]
            mass = Config.mass[key+1]
            jointpos = jointsPos[i]
            parentpos = jointsPos[parentIdx[i]]
            if name ==  'lankle':
                link = Link(name+'_rx')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = Link(name+'_ry')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = Link(name+'_rz')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = Link(name)
                link.iners.append(Inertial(Config.lankle_offset,[0,0,0],mass,Config.iner))
                geo = Geometry('box', Config.ankle_size[0], Config.ankle_size[1], Config.ankle_size[2])
                link.colls.append(Collision(Config.lankle_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'rankle':
                link = Link(name+'_rx')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = Link(name+'_ry')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = Link(name+'_rz')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = Link(name)
                link.iners.append(Inertial(Config.rankle_offset,[0,0,0],mass,Config.iner))
                geo = Geometry('box', Config.ankle_size[0], Config.ankle_size[1], Config.ankle_size[2])
                link.colls.append(Collision(Config.rankle_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'lowerback':
                link = Link(name+'_rx')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = Link(name+'_ry')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = Link(name+'_rz')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = Link(name)
                link.iners.append(Inertial(Config.lowerback_offset,[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision(Config.lowerback_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'upperback':
                link = Link(name+'_rx')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = Link(name+'_ry')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = Link(name+'_rz')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = Link(name)
                link.iners.append(Inertial(Config.upperback_offset,[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision(Config.upperback_offset,[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'chest':
                link = Link(name+'_rx')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rx', 'revolute', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = Link(name+'_ry')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_ry', 'revolute', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = Link(name+'_rz')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rz', 'revolute', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = Link(name)
                link.iners.append(Inertial(Config.chest_offset,[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision(Config.chest_offset+Config.chest_det,[0,0,0],name+'0',geo))
                link.colls.append(Collision(Config.chest_offset-Config.chest_det,[0,0,0],name+'1',geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'lwrist':
                link = Link(name+'_rx')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rx', 'fixed', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = Link(name+'_ry')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_ry', 'fixed', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = Link(name+'_rz')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rz', 'fixed', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = Link(name)
                link.iners.append(Inertial([weigth,0,0],[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision([weigth,0,0],[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)
            elif name == 'rwrist':
                link = Link(name+'_rx')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rx', 'fixed', Config.parentname[key+1], name+'_rx', jointpos-parentpos, [0,0,0], [1,0,0], limit)
                joint.writeFile(file)

                link = Link(name+'_ry')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_ry', 'fixed', name+'_rx', name+'_ry', [0,0,0], [0,0,0], [0,1,0], limit)
                joint.writeFile(file)

                link = Link(name+'_rz')
                link.iners.append(Inertial([0,0,0],[0,0,0],Config.dotmass,Config.iner))
                link.writeFile(file)
                limit = Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
                joint = Joint(name+'_rz', 'fixed', name+'_ry', name+'_rz', [0,0,0], [0,0,0], [0,0,1], limit)
                joint.writeFile(file)

                link = Link(name)
                link.iners.append(Inertial([-1*weigth,0,0],[0,0,0],mass,Config.iner))
                geo = Geometry(shape,weigth)
                link.colls.append(Collision([-1*weigth,0,0],[0,0,0],name,geo))
                link.writeFile(file)
                joint = Joint(name, 'fixed', name+'_rz', name, [0,0,0], [0,0,0], [1,0,0])
                joint.writeFile(file)

    write_end(file)
    file.close()

if __name__ == '__main__':
    config = {
        'pklPath' : R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-physics\00000.pkl',
        'urdfPath' : R'\\105.1.1.112\e\Human-Data-Physics-v1.0/demo3.urdf',
        'transPath' : R'\\105.1.1.112\e\Human-Data-Physics-v1.0/demo3.txt',
        'isZero' : True
    }
    smpl2Urdf(config)
    config = {
        'pklPath' : R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-physics\00000.pkl',
        'urdfPath' : R'\\105.1.1.112\e\Human-Data-Physics-v1.0/demo3Dof.urdf',
        'isZero' : False
    }
    smpl2UrdfDof(config)