import numpy as np
def read_point(file_name):
    verts = []
    f = open(file_name)
    line = f.readline()
    while line:
        line_data = line.split()
        verts.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
        #color.append([float(line_data[4]),float(line_data[5]),float(line_data[6])])
        line = f.readline()
    f.close()
    return verts#,color
verts = np.array(read_point('human36Joints.obj'))

RRR = np.loadtxt('Human36RT.txt')

# RRR = np.array([
#     [0.99994531,  0.01038064,  0.00127162, 0.00730746],
#     [-0.01038064,  0.97037788,  0.24136904, 1.38704965],
#     [0.00127162, -0.24136904,  0.97043257, 5.57668113],
#     [0.0, 0.0, 0.0, 1.0]
#     ])
verts_new = np.dot(RRR[:3,:3], verts.T) + RRR[:3,3][:,None]

def write_obj(verts):
    f = open('jointsHuman.obj','w')
    for v in verts:
        f.write('v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
    f.close()
write_obj(verts_new.T)


