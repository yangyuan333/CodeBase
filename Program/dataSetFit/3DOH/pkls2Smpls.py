from genericpath import exists
import glob
import os
import sys

from pathy import shutil
sys.path.append('./')
import pickle as pkl

from utils.obj_utils import read_obj, write_obj
from utils.smpl_utils import pkl2Smpl
meshData = read_obj(R'./data/smpl/template.obj')

def pkls2Smpls(config):
    if os.path.exists(config['savePath']):
        shutil.rmtree(config['savePath'])
    os.makedirs(config['savePath'],exist_ok=True)

    for path in glob.glob(os.path.join(config['pklPath'],'*')):
        vs,js = pkl2Smpl(path)
        meshData.vert = vs
        write_obj(os.path.join(config['savePath'],os.path.basename(path).split('.')[0]+'.obj'), meshData)

if __name__ == '__main__':
    config = {
        'pklPath':R'E:\Human-Data-Physics-v1.0\3DOH-physics',
        'savePath':R'E:\Human-Data-Physics-v1.0\tem'
    }
    pkls2Smpls(config)