from calendar import c
import glob
import os
import sys
sys.path.append('./')
import pickle as pkl
import shutil

from utils.smpl_utils import SMPLModel
smplModel = SMPLModel()
from utils.obj_utils import read_obj,write_obj
meshData = read_obj(R'./data/smpl/template.obj')

def copyFiles(config):
    for path in glob.glob(os.path.join(config['rootPath'],'*')):
        shutil.copyfile(
            os.path.join(path,'000.pkl'),
            os.path.join(config['savePath'],os.path.basename(path)+'.pkl')
        )

if __name__ == '__main__':
    config = {
        'rootPath':R'\\105.1.1.112\e\Human-Data-Physics-v1.0\3DOH-GT\params_米制Y轴\0006',
        'savePath':R'\\105.1.1.112\e\Human-Data-Physics-v1.0\tem\pkl'
    }
    copyFiles(config)