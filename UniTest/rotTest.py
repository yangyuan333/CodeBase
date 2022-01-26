import sys
sys.path.append('./')
from utils.rotate_utils import *

mat = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])
r = R.from_matrix(mat)
print(r.as_rotvec())
r = R.from_rotvec([0,0,0])
print(r.as_matrix())