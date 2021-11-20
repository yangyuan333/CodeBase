import pybullet as p
import time
import math
import pybullet_data
import sys
sys.path.append('./')

cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(1. / 120.)
logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
#useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)
#p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
#disable rendering during creation.
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

shift = [0, 0, 0]
meshScale = [1.0,1.0,1.0]
#the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName="./data/urdf/0000.obj",
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                          collisionFramePosition=shift,
                                          meshScale=meshScale)
p.createMultiBody(
    baseMass=0,
    baseInertialFramePosition=[0, 0, 0],
    baseCollisionShapeIndex=collisionShapeId,
    basePosition=[0, 0, 0],
    useMaximalCoordinates=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.stopStateLogging(logId)
p.setGravity(0, -10, 0)
# p.setRealTimeSimulation(1)

colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
currentColor = 0

sphereRadius = 0.1
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
# colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])

mass = 1
visualShapeId = -1

sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, [1,-1,1], [0,0,0,1])

while (1):
    p.setRealTimeSimulation(1)