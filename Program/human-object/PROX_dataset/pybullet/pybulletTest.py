import pybullet as p
import time
import pybullet_data
import math
import sys
sys.path.append('./')

def pybulletInit():
    physicsCilent = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,-9.8,0)
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
    p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

if __name__ == '__main__':
    
    pybulletInit()

    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF(
        R"./data/urdf/vicon_final.urdf",
        startPos, 
        startOrientation,
        globalScaling=1.0,
        )

    sphereRadius = 0.15
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    visualShapeId = -1
    mass = 1
    sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, [-2,1.0,0], [0, 0, 0, 1])

    # boxId = p.loadURDF(
    #     R"./data/urdf/smpl.urdf",
    #     [0,0,0], 
    #     [0,0,0,1],
    #     globalScaling=1.0,
    # )

    smpl_visual_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=R"./data/urdf/000_smpl_final.obj",
        meshScale=1.0)
    smpl_collision_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,  
        fileName=R"./data/urdf/000_smpl_final.obj",
        meshScale=1.0)
    smpl = p.createMultiBody(
        baseMass=60.0,
        baseCollisionShapeIndex=smpl_collision_id,
        baseVisualShapeIndex=smpl_visual_id,
        basePosition=[5,1,0],
        flags=p.URDF_USE_SELF_COLLISION)

    flag = False
    while (p.isConnected()):
        if ~flag:
            qKey = ord('q')
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
                flag = True
        if flag:
            p.stepSimulation()