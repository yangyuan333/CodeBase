import pybullet as p
import time
import pybullet_data
import math
import sys
sys.path.append('./')

def pybulletInit():
    physicsCilent = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,-0.1,0)
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
    p.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

if __name__ == '__main__':
    pybulletInit()
    boxId = p.loadURDF(
        "duck_vhacd.urdf",
        [0,0,0], 
        [0,0,0,1],
        globalScaling=20.0,
    )
    # boxId2 = p.loadURDF(
    #     "duck_vhacd.urdf",
    #     [0.05,1,0], 
    #     [0,0,0,1],
    #     globalScaling=5.0,
    # )
    sphereRadius = 0.1
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)

    visualShapeId = -1
    mass = 1
    sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, [0.2,0.8,0], [0, 0, 0, 1])

    flag = False
    while (p.isConnected()):
        if ~flag:
            qKey = ord('q')
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
                flag = True
        if flag:
            p.stepSimulation()