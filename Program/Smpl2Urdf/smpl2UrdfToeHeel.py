from ntpath import join
import pybullet as p
import pybullet_data
import math
import sys
sys.path.append('./')
from utils.smpl_utils import SMPLModel
import torch
import numpy as np
import utils.urdf_utils as urdf_utils
from utils.rotate_utils import *
import pickle

class Config(object):
    urdfPath = r'./data/temdata/shape-ToeHeel.urdf'
    transPath = r'./data/temdata/shape-ToeHeel.txt'
    iner = 0.001
    ankle_size = [0.0875,0.06,0.185]
    lankle_offset = [0.01719,-0.06032,0.02617]
    rankle_offset = [-0.01719,-0.06032,0.02617]
    lowerback_offset = [0.0,0.05,0.013]
    upperback_offset = [0.0,0.02246,0.00143]
    chest_offset = np.array([0.0,0.057,-0.00687])
    chest_det = np.array([0.045,0,0])
    upperneck_length = 0.035
    # ankle_offset = [0,0,0]
    # lankle_offset = [0,0,0]
    # rankle_offset = [0,0,0]
    mass = [5,5,3,1,5,3,1,5,5,8,0.5,3,1,2,1,0.5,1,2,1,0.5] # 所有link
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
    limit = {
        'effort':1000.0,
        'lower':-3.14,
        'upper':3.14,
        'velocity':0.5
    }

smplModel = SMPLModel()
# smpl_vs, smpl_js = smplModel(betas=torch.tensor(np.zeros((1, 10)).astype(np.float32)), thetas=torch.tensor(np.zeros((1, 72)).astype(np.float32)), trans=torch.tensor(np.zeros((1, 3)).astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
# smpl_vs, smpl_js = smplModel(betas=torch.tensor(np.random.rand(10).astype(np.float32)[None,:]), thetas=torch.tensor(np.zeros((1, 72)).astype(np.float32)), trans=torch.tensor(np.zeros((1, 3)).astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
#betas = [-0.12818438, -1.0672331, -0.010120345, 1.4188042, 0.47162628, -0.22455935, -0.6319933, 0.32396683, 0.22103898, -0.18049102]
# betas = [
#     0.00303079, -0.06827933,  0.13632792,  0.15463829,  0.02087251,
#     0.04367678, -0.00264158, -0.02082669,  0.01223242, -0.01340746
# ]

# pklPath = r'H:\YangYuan\Code\phy_program\CodeBase\data\GTA_2307_BASE.pkl'
pklPath = r'./data/temdata/0000000013.pkl'
with open(pklPath, 'rb') as file:
    data = pickle.load(file)    
betas = data['person00']['betas'][0]
# betas = [
#     0.0, 0.0, 0.0,  0.0,  0.0,
#     0.0, 0.0, 0.0,  0.0, -0.0
# ]
smpl_vs, smpl_js = smplModel(betas=torch.tensor(np.array(betas).astype(np.float32)[None,:]), thetas=torch.tensor(np.zeros((1, 72)).astype(np.float32)), trans=torch.tensor(np.zeros((1, 3)).astype(np.float32)), scale=torch.tensor([1]), gR=None, lsp=False)
smpl_js = smpl_js.squeeze(0).numpy() # 24*3
smpl_vs = smpl_vs.squeeze(0).numpy()
urdfPath = Config.urdfPath
parentIdx = [
    -1,0,0,0,1,2,3,4,5,6,7,8,
    9,9,9,12,13,14,16,17,18,19,20,21] # 24,smpl中每个joint的父joint
childIdx = [
    -1,4,5,6,7,8,9,10,11,-1,-1,-1,
    15,16,17,-1,18,19,20,21,22,23,-1,-1] # 24,smpl中每个joint的子joint
jointsPos = smpl_js # 24*3
jointInUrdfIdx = [1,4,7,2,5,8,3,6,9,12,15,13,16,18,20,14,17,19,21] # 每个数代表smpl中的节点序号
#jointInUrdfIdx = [1,4,7]

file = open(urdfPath, 'w')
urdf_utils.write_start(file, 'amass')

#root link
root_link = urdf_utils.Link('root')
print(-jointsPos[0])
np.savetxt(Config.transPath, -jointsPos[0])
# root_link.iners.append(urdf_utils.Inertial([0,0,0], [0,0,0], Config.mass[0], Config.iner))
root_link.iners.append(urdf_utils.Inertial(-jointsPos[0], [0,0,0], Config.mass[0], Config.iner))
# root_link.iners.append(urdf_utils.Inertial([0,0,0], [0,0,0], Config.mass[0], Config.iner))
root_link.colls.append(urdf_utils.Collision([0.00354, 0.065, -0.03107], [0, 1.5708, 0], 'collision_0_root', urdf_utils.Geometry('sphere', 0.05, 0.115)))
root_link.colls.append(urdf_utils.Collision([-0.05769, -0.02577, -0.0174], [0, 0, 0], 'collision_1_root', urdf_utils.Geometry('sphere', 0.075)))
root_link.colls.append(urdf_utils.Collision([0.06735, -0.02415, -0.0174], [0, 0, 0], 'collision_2_root', urdf_utils.Geometry('sphere', 0.075)))
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

        link = urdf_utils.Link(name)
        link.iners.append(urdf_utils.Inertial([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]],[0,0,0],mass,Config.iner))
        # link.iners.append(urdf_utils.Inertial([0, 0, parentpos[2]-jointpos[2]],[0,0,0],mass,Config.iner))

        rotvec = np.array([0,1,0])
        r = CalRotFromVecs(np.array([0,0,1]), rotvec)
        if shape == 'capsule':
            geo = urdf_utils.Geometry('capsule', weigth, length)
        elif shape == 'box':
            geo = urdf_utils.Geometry('box', weigth, weigth, length+1.6*weigth)
        link.colls.append(urdf_utils.Collision([0, -(length+3*weigth)/2, parentpos[2]-jointpos[2]], r.as_euler('xyz'), name, geo))
        # link.colls.append(urdf_utils.Collision([0, 0, parentpos[2]-jointpos[2]], r.as_euler('xyz'), name, geo))

        link.writeFile(file)

        joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])

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

        link = urdf_utils.Link(name)
        # link.iners.append(urdf_utils.Inertial([0, (length+2*weigth)/2, 0],[0,0,0],mass,Config.iner))
        link.iners.append(urdf_utils.Inertial([0, 0, 0],[0,0,0],mass,Config.iner))
        rotvec = np.array([0,1,0])
        r = CalRotFromVecs(np.array([0,0,1]), rotvec)
        if shape == 'capsule':
            geo = urdf_utils.Geometry('capsule', weigth, length)
        elif shape == 'box':
            geo = urdf_utils.Geometry('box', weigth, weigth, length+1.6*weigth)
        # link.colls.append(urdf_utils.Collision([0, (length+2*weigth)/2, 0], r.as_euler('xyz'), name, geo))
        link.colls.append(urdf_utils.Collision([0, 0, 0], r.as_euler('xyz'), name, geo))
        link.writeFile(file)

        joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])

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

        link = urdf_utils.Link(name)
        link.iners.append(urdf_utils.Inertial((childpos-jointpos)/2.0,[0,0,0],mass,Config.iner))

        rotvec = np.array(childpos-jointpos)
        r = CalRotFromVecs(np.array([0,0,1]), rotvec)
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

        joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])

        joint.writeFile(file)

    else:
        weigth = Config.weigh[key+1]
        shape = Config.shape[key+1]
        name = Config.name[key+1]
        mass = Config.mass[key+1]
        jointpos = jointsPos[i]
        parentpos = jointsPos[parentIdx[i]]
        if name ==  'lankle':
            link = urdf_utils.Link(name)
            link.iners.append(urdf_utils.Inertial(Config.lankle_offset,[0,0,0],mass,Config.iner))
            geo = urdf_utils.Geometry('box', Config.ankle_size[0], Config.ankle_size[1], Config.ankle_size[2])
            link.colls.append(urdf_utils.Collision(Config.lankle_offset,[0,0,0],name,geo))
            link.writeFile(file)
            joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
            joint.writeFile(file)
        elif name == 'rankle':
            link = urdf_utils.Link(name)
            link.iners.append(urdf_utils.Inertial(Config.rankle_offset,[0,0,0],mass,Config.iner))
            geo = urdf_utils.Geometry('box', Config.ankle_size[0], Config.ankle_size[1], Config.ankle_size[2])
            link.colls.append(urdf_utils.Collision(Config.rankle_offset,[0,0,0],name,geo))
            link.writeFile(file)
            joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
            joint.writeFile(file)
        elif name == 'lowerback':
            link = urdf_utils.Link(name)
            link.iners.append(urdf_utils.Inertial(Config.lowerback_offset,[0,0,0],mass,Config.iner))
            geo = urdf_utils.Geometry(shape,weigth)
            link.colls.append(urdf_utils.Collision(Config.lowerback_offset,[0,0,0],name,geo))
            link.writeFile(file)
            joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
            joint.writeFile(file)
        elif name == 'upperback':
            link = urdf_utils.Link(name)
            link.iners.append(urdf_utils.Inertial(Config.upperback_offset,[0,0,0],mass,Config.iner))
            geo = urdf_utils.Geometry(shape,weigth)
            link.colls.append(urdf_utils.Collision(Config.upperback_offset,[0,0,0],name,geo))
            link.writeFile(file)
            joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
            joint.writeFile(file)
        elif name == 'chest':
            link = urdf_utils.Link(name)
            link.iners.append(urdf_utils.Inertial(Config.chest_offset,[0,0,0],mass,Config.iner))
            geo = urdf_utils.Geometry(shape,weigth)
            link.colls.append(urdf_utils.Collision(Config.chest_offset+Config.chest_det,[0,0,0],name+'0',geo))
            link.colls.append(urdf_utils.Collision(Config.chest_offset-Config.chest_det,[0,0,0],name+'1',geo))
            link.writeFile(file)
            joint = urdf_utils.Joint(name, 'spherical', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
            joint.writeFile(file)
        elif name == 'lwrist':
            link = urdf_utils.Link(name)
            link.iners.append(urdf_utils.Inertial([weigth,0,0],[0,0,0],mass,Config.iner))
            geo = urdf_utils.Geometry(shape,weigth)
            link.colls.append(urdf_utils.Collision([weigth,0,0],[0,0,0],name,geo))
            link.writeFile(file)
            joint = urdf_utils.Joint(name, 'fixed', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
            joint.writeFile(file)
        elif name == 'rwrist':
            link = urdf_utils.Link(name)
            link.iners.append(urdf_utils.Inertial([-1*weigth,0,0],[0,0,0],mass,Config.iner))
            geo = urdf_utils.Geometry(shape,weigth)
            link.colls.append(urdf_utils.Collision([-1*weigth,0,0],[0,0,0],name,geo))
            link.writeFile(file)
            joint = urdf_utils.Joint(name, 'fixed', Config.parentname[key+1], name, jointpos-parentpos, [0,0,0], [1,0,0])
            joint.writeFile(file)

r = 0.005
ltoe_link = urdf_utils.Link('ltoe')
ltoe_link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],0.000001,Config.iner))
geo = urdf_utils.Geometry('cylinder', r, 1*Config.ankle_size[0])
ltoe_link.colls.append(urdf_utils.Collision([0,0,0], [0,1.57,0], 'ltoe', geo))
ltoe_link.writeFile(file)
limit = urdf_utils.Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
ltoe_joint = urdf_utils.Joint('ltoe', 'revolute', 'lankle', 'ltoe', [Config.lankle_offset[0], Config.lankle_offset[1]-Config.ankle_size[1]/2-r, Config.lankle_offset[2]+Config.ankle_size[2]/2-r], [0,0,0], [0,1,0], limit)
ltoe_joint.writeFile(file)

lheel_link = urdf_utils.Link('lheel')
lheel_link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],0.000001,Config.iner))
geo = urdf_utils.Geometry('cylinder', r, 1*Config.ankle_size[0])
lheel_link.colls.append(urdf_utils.Collision([0,0,0], [0,1.57,0], 'lheel', geo))
lheel_link.writeFile(file)
lheel_joint = urdf_utils.Joint('lheel', 'revolute', 'ltoe', 'lheel', [0,0,-(1*Config.ankle_size[2]-2*r)], [0,0,0], [0,1,0],limit)
lheel_joint.writeFile(file)

rtoe_link = urdf_utils.Link('rtoe')
rtoe_link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],0.000001,Config.iner))
geo = urdf_utils.Geometry('cylinder', r, 1*Config.ankle_size[0])
rtoe_link.colls.append(urdf_utils.Collision([0,0,0], [0,1.57,0], 'rtoe', geo))
rtoe_link.writeFile(file)
limit = urdf_utils.Limit(Config.limit['effort'], Config.limit['lower'], Config.limit['upper'], Config.limit['velocity'])
rtoe_joint = urdf_utils.Joint('rtoe', 'revolute', 'rankle', 'rtoe', [Config.rankle_offset[0], Config.rankle_offset[1]-Config.ankle_size[1]/2-r, Config.rankle_offset[2]+Config.ankle_size[2]/2-r], [0,0,0], [0,1,0], limit)
rtoe_joint.writeFile(file)

rheel_link = urdf_utils.Link('rheel')
rheel_link.iners.append(urdf_utils.Inertial([0,0,0],[0,0,0],0.000001,Config.iner))
geo = urdf_utils.Geometry('cylinder', r, 1*Config.ankle_size[0])
rheel_link.colls.append(urdf_utils.Collision([0,0,0], [0,1.57,0], 'rheel', geo))
rheel_link.writeFile(file)
rheel_joint = urdf_utils.Joint('rheel', 'revolute', 'rtoe', 'rheel', [0,0,-(1*Config.ankle_size[2]-2*r)], [0,0,0], [0,1,0],limit)
rheel_joint.writeFile(file)

urdf_utils.write_end(file)
file.close()