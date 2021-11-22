import pybullet as p
import time
import pybullet_data
import sys
sys.path.append('./')
# 连接物理引擎
physicsCilent = p.connect(p.GUI)

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置环境重力加速度
p.setGravity(0,-0.1,0)
import math
# 加载URDF模型，此处是加载蓝白相间的陆地
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)

# 加载机器人，并设置加载的机器人的位姿
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
#boxId = p.loadURDF("./data/urdf/plane0000.urdf",startPos, startOrientation,globalScaling=10.0)
boxId = p.loadURDF(
    r"H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\urdf\plane0000.urdf",
    startPos, 
    startOrientation,
    globalScaling=1.0,
    )

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.setGravity(0, -10, 0)
# p.setRealTimeSimulation(1)

colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
currentColor = 0

sphereRadius = 0.1
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)

visualShapeId = -1
mass = 1
sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, [0,0.4,-0.2], [0, 0, 0, 1])
# sphereUid1 = p.createMultiBody(mass, colSphereId, visualShapeId, [0.05,1,0], [0, 0, 0, 1])
while (p.isConnected()):
    p.stepSimulation()
    # p.setRealTimeSimulation(1)