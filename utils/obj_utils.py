# def read_obj(file_name):
#     verts = []
#     faces = []
#     f = open(file_name)
#     line = f.readline()
#     while line:
#         line_data = line.split()
#         if line_data.__len__():
#             if line_data[0] == 'v':
#                 verts.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
#             elif line_data[0] == 'f':
#                 faces.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])           
#         line = f.readline()
#     f.close()
#     return verts, faces

# def write_obj(file_name, meshData):
#     f = open(file_name, 'w')
#     for data in verts:
#         f.write('v ' + str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + '\n')
#     if faces is None:
#         f.close()
#         return
#     for face in faces:
#         f.write('f ' + str(int(face[0])) + ' ' + str(int(face[1])) + ' ' + str(int(face[2])) + '\n')
#     f.close()

class MeshData(object):
    def __init__(self):
        self.vert = []
        self.face = []
        self.vn = []
        self.color = []

def read_obj(file_name):
    meshData = MeshData()
    f = open(file_name)
    line = f.readline()
    while line:
        line_data = line.split()
        if line_data.__len__():
            if line_data[0] == 'v':
                meshData.vert.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
                if line_data.__len__() == 7:
                    meshData.color.append([float(line_data[4]),float(line_data[5]),float(line_data[6])])
            elif line_data[0] == 'f':
                meshData.face.append([int(line_data[1].split('//')[0]),int(line_data[2].split('//')[0]),int(line_data[3].split('//')[0])])           
            elif line_data[0] == 'vn':
                meshData.vn.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
        line = f.readline()
    f.close()
    return meshData

def write_obj(file_name, meshData):
    f = open(file_name, 'w')
    if meshData.color.__len__():
        for v, c in zip(meshData.vert, meshData.color):
            f.write('v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + ' ' + str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + '\n')
    else:
        for v in meshData.vert:
            f.write('v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
    if meshData.vn.__len__():
        for vn in meshData.vn:
            f.write('vn ' + str(vn[0]) + ' ' + str(vn[1]) + ' ' + str(vn[2]) + '\n')
    if meshData.face.__len__():
        for face in meshData.face:
            f.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')

def Scaling_obj(file_name, save_file):
    f= open(file_name)
    strData = []
    line = f.readline()
    while(line):
        line_data = line.split()
        if line_data.__len__():
            if line_data[0] == 'v':
                vert = [float(line_data[1]),float(line_data[2]),float(line_data[3])]
                line = 'v ' + str(vert[0]/100) + ' ' + str(vert[1]/100) + ' ' + str(vert[2]/100) + '\n'
        strData.append(line)
        line = f.readline()
    f.close()
    file = open(save_file, 'w')
    for data in strData:
        file.write(data)
    file.close()

if __name__ == '__main__':
    Scaling_obj(
        r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf\0000.obj',
        r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf\0000m.obj'
        )
    Scaling_obj(
        r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf\0034.obj',
        r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf\0034m.obj'
        )
    Scaling_obj(
        r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf\0052.obj',
        r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master\data\graphics\physdata\urdf\0052m.obj'
        )

# if __name__ == '__main__':
#     objPath = r'H:\YangYuan\Code\phy_program\CodeBase\utils\GTAIM-case1.obj'
#     meshdata = read_obj(objPath)
#     print(0)
#     write_obj(r'H:\YangYuan\Code\phy_program\CodeBase\utils\GTAIM-case2.obj', meshdata)