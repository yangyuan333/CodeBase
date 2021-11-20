import os
import glob
import json

keyspointsPath = glob.glob(os.path.join(r'H:\YangYuan\Code\phy_program\CodeBase\GTA\GTA-test\keypoints\TS1\Camera00', '*'))
for keyPath in keyspointsPath:
    with open(keyPath, 'rb') as file:
        keyData = json.load(file)
    print(1)