import pybullet as p
import pybullet_data
import time
import math

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.setTimeStep(1. / 120.)
# useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0]) # 绕x轴旋转-90度
p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)


shift = [0, 0, 0]
meshScale = [1, 1, 1]
obj_name = "human36test.obj"
# the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=obj_name,
                                    rgbaColor=[1, 0, 0, 1],
                                    specularColor=[0.4, .4, 0],
                                    visualFramePosition=shift,
                                    meshScale=meshScale)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName=obj_name,
                                          collisionFramePosition=shift,
                                          meshScale=meshScale)

p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0, 0, 4],
    useMaximalCoordinates=True
)

while (1):
    time.sleep(1./240.)

