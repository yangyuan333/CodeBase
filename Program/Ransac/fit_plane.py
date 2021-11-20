import numpy as np

data = np.loadtxt('Human36V3.txt')
print(data.shape)

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

for i in range(data.shape[0]):
    A11 += (data[i][0]*data[i][0])
    A12 += (data[i][0]*data[i][1])
    A13 += (data[i][0])
    A21 += (data[i][0]*data[i][1])
    A22 += (data[i][1]*data[i][1])
    A23 += (data[i][1])
    A31 += (data[i][0])
    A32 += (data[i][1])
    A33 += (1)
    S0 += (data[i][0]*data[i][2])
    S1 += (data[i][1]*data[i][2])
    S2 += (data[i][2])

a = np.linalg.solve([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]],[S0,S1,S2])
print(a)

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_point(file_name):
    verts = []
    color = []
    f = open(file_name)
    line = f.readline()
    while line:
        line_data = line.split()
        verts.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
        #color.append([float(line_data[4]),float(line_data[5]),float(line_data[6])])
        line = f.readline()
    f.close()
    return verts#,color

verts = read_point('human36JointsTest.obj')
print(verts.__len__())

min_x = int(min(np.array(verts)[:,0]))
max_x = int(max(np.array(verts)[:,0]))
min_y = int(min(np.array(verts)[:,1]))
max_y = int(max(np.array(verts)[:,1]))

sample_num = 10000
zs = []
xs = []
ys = []
for i in range(sample_num):
    x = np.random.uniform(min_x,max_x)
    y = np.random.uniform(min_y, max_y)
    z = a[0]*x + a[1]*y + a[2]
    xs.append(x)
    ys.append(y)
    zs.append(z)
f = open('human36Plane.txt','w')
for x,y,z in zip(xs,ys,zs):
    f.write('v ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
f.close()


Visual = False
if Visual:
    min_x = min(np.array(verts)[:,0])
    max_x = max(np.array(verts)[:,0])
    min_y = min(np.array(verts)[:,1])
    max_y = max(np.array(verts)[:,1])
    fig = plt.figure(figsize=(12, 8),
                     facecolor='lightyellow'
                    )
    ax = fig.gca(fc='whitesmoke',
                 projection='3d'
                )
    x = np.linspace(int(min_x)-1, int(max_x)+1, 100)
    y = np.linspace(int(min_y)-1, int(max_y)+1, 100)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X,
                    Y,
                    Z=X*a[0]+Y*a[1]+a[2],
                    color='g',
                    alpha=0.6
                   )
    ax.scatter(np.array(verts)[:,0],np.array(verts)[:,1],np.array(verts)[:,2],c='r',s=2)
    plt.show()

f_n = np.array([a[0],a[1],-1])
f_n = -f_n / np.linalg.norm(f_n)
print('f_n', f_n)
n = np.array([0,-1,0])
axis = np.cross(n, f_n)
axis = axis / np.linalg.norm(axis)
print('axis',axis)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot([0,f_n[0]],[0,f_n[1]],[0,f_n[2]],c='r')
# ax.plot([0,n[0]],[0,n[1]],[0,n[2]],c='g')
# ax.plot([0,axis[0]],[0,axis[1]],[0,axis[2]],c='b')
# plt.show()

import math
sita = -math.acos(np.dot(f_n, n))
print(sita)
print(sita*180/3.1415)
import cv2
R, j = cv2.Rodrigues(axis*sita)
print(R)
print('trans_v',np.dot(R,f_n))


verts = np.array(verts).transpose()
print(verts.shape)
verts_new = R @ verts
verts_new = verts_new.transpose()

point_sample = R @ np.array([0,0,a[2]])
print(point_sample)
verts_new[:,1] = verts_new[:,1] - point_sample[1]

min_x = min(np.array(verts_new)[:,0])
max_x = max(np.array(verts_new)[:,0])
min_y = min(np.array(verts_new)[:,1])
max_y = max(np.array(verts_new)[:,1])
# center_x = (max_x + min_x) / 2
# center_y = (max_y + min_y) / 2
# verts_new[:,0] -= center_x
# verts_new[:,1] -= center_y

def write_obj(verts):
    f = open('new_vert.txt','w')
    for v in verts:
        f.write('v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
    f.close()
write_obj(verts_new)

Visual_points = False
if Visual_points:
    min_x = min(np.array(verts_new)[:,0])
    max_x = max(np.array(verts_new)[:,0])
    min_y = min(np.array(verts_new)[:,1])
    max_y = max(np.array(verts_new)[:,1])
    fig = plt.figure(figsize=(12, 8),
                     facecolor='lightyellow'
                    )
    ax = fig.gca(fc='whitesmoke',
                 projection='3d'
                )
    x = np.linspace(int(min_x)-1, int(max_x)+1, 100)
    y = np.linspace(int(min_y)-1, int(max_y)+1, 100)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X,
                    Y,
                    Z=X*a[0]+Y*a[1]+a[2],
                    color='g',
                    alpha=0.6
                   )
    ax.scatter(np.array(verts_new)[:,0],np.array(verts_new)[:,1],np.array(verts_new)[:,2],c='r',s=2)
    plt.show()