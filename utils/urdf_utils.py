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