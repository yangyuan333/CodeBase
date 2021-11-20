import pybullet as p
import time
import math
import pybullet_data

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
meshScale = [0.3,0.3,0.3]
#the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName="vn_f_xishu.obj",
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                          collisionFramePosition=shift,
                                          meshScale=meshScale)

rangex = 1
rangey = 1
for i in range(rangex):
  for j in range(rangey):
    p.createMultiBody(baseMass=0,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      basePosition=[0, 0, 0],
                      #baseOrientation=p.getQuaternionFromEuler([0,0,2.2]),
                      useMaximalCoordinates=True)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.stopStateLogging(logId)
p.setGravity(0, -10, 0)
p.setRealTimeSimulation(1)

colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
currentColor = 0


sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

mass = 1
visualShapeId = -1

link_Masses = [1]
linkCollisionShapeIndices = [colBoxId]
linkVisualShapeIndices = [-1]
linkPositions = [[0, 0.11, 0]]
linkOrientations = [[0, 0, 0, 1]]
linkInertialFramePositions = [[0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1]]
indices = [0]
jointTypes = [p.JOINT_REVOLUTE]
axis = [[0, 1, 0]]

for i in range(3):
    for j in range(3):
        for k in range(3):
            basePosition = [
                i * 5 * sphereRadius, j * 5 * sphereRadius+2, k * 5 * sphereRadius
            ]
            baseOrientation = [0, 0, 0, 1]
            if (k & 2):
                sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, basePosition,
                                              baseOrientation)
            else:
                sphereUid = p.createMultiBody(mass,
                                              colBoxId,
                                              visualShapeId,
                                              basePosition,
                                              baseOrientation,
                                              linkMasses=link_Masses,
                                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                                              linkVisualShapeIndices=linkVisualShapeIndices,
                                              linkPositions=linkPositions,
                                              linkOrientations=linkOrientations,
                                              linkInertialFramePositions=linkInertialFramePositions,
                                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                                              linkParentIndices=indices,
                                              linkJointTypes=jointTypes,
                                              linkJointAxis=axis)

            p.changeDynamics(sphereUid,
                             -1,
                             spinningFriction=0.001,
                             rollingFriction=0.001,
                             linearDamping=0.0)
            for joint in range(p.getNumJoints(sphereUid)):
                p.setJointMotorControl2(sphereUid, joint, p.VELOCITY_CONTROL, targetVelocity=1, force=10)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, -10, 0)
p.setRealTimeSimulation(1)

p.getNumJoints(sphereUid)
for i in range(p.getNumJoints(sphereUid)):
    p.getJointInfo(sphereUid, i)

while (p.isConnected()):
    keys = p.getKeyboardEvents()

    # print(keys)
    # getCameraImage note: software/TinyRenderer doesn't render/support heightfields!
    # p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    time.sleep(0.01)

while (1):
  time.sleep(1./240.)