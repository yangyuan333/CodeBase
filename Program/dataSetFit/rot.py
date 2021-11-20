import numpy as np
def read_point(file_name):
    verts = []
    color = []
    f = open(file_name)
    line = f.readline()
    while line:
        line_data = line.split()
        if line_data[0] == 'v':
            verts.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
        #color.append([float(line_data[4]),float(line_data[5]),float(line_data[6])])
        line = f.readline()
    f.close()
    return verts#,color

verts = read_point('template.obj')
from scipy.spatial.transform import Rotation as R
r = R.from_rotvec([-0.28871483,-2.3812518,-0.7716103])
rot = r.as_matrix()
verts = np.array(verts).T
v_new = np.dot(rot, verts).T

f = open('t1.obj', 'w')
for data in v_new:
    f.write('v ' + str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + '\n')
f.close()



