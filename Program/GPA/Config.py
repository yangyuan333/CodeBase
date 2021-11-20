import os
import json

class GPATool(object):
    def __init__(self, annotRootPath):
        self.imgName2idx = {}
        with open(os.path.join(annotRootPath, 'xyz_gpa12_cntind_world_cams.json'), 'r') as file:
            annotDatas = json.load(file)
        idx = 0
        for imageData, annotData in zip(annotDatas['images'], annotDatas['annotations']):
            imageName = os.path.basename(imageData['file_name'])[:(-4)]
            self.imgName2idx[imageName] = idx
            idx += 1
        self.annotDatas = annotDatas