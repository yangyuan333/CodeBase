from scipy import integrate
import numpy as np
import math
def f(x, y):
    return math.tan(x*x+y*y)

v, err = integrate.dblquad(f, 0, math.pi/3, lambda x: 0, lambda x: math.pi/6)
print(v)