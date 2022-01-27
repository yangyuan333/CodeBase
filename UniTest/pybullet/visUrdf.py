import pybullet as p
import pybullet_data
import math
import sys
sys.path.append('./')

def showURDF(config):
    physicsCilent = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
    planeId = p.loadURDF("plane.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
    physcapHuman = p.loadURDF(
        fileName = config['urdfPath'],
        basePosition = [0,0,0],
        baseOrientation = p.getQuaternionFromEuler([0,0,0]),
        globalScaling = 1,
        useFixedBase = True,
        flags = p.URDF_MAINTAIN_LINK_ORDER
    )

    while(p.isConnected()):
        pass

if __name__ == '__main__':
    config = {
        'urdfPath':R'\\105.1.1.112\e\Human-Data-Physics-v1.0\demo3Dof.urdf'
    }
    showURDF(config)