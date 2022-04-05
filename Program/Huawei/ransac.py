import numpy as np
import random

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        print(s)
        print('estimate:', m,)
        print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[:10, :10]
        return xx, yy, (-d - a * xx - b * yy) / c

    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3

    # test data
    xyzs = np.random.random((n, 3)) * 10
    xyzs[:50, 2:] = xyzs[:50, :1]

    import sys
    sys.path.append('./')
    from utils.obj_utils import read_obj,write_obj,MeshData
    meshData = read_obj('./data/temdata/results/plane0.obj')
    xyzs = np.array(meshData.vert)
    ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

    # RANSAC
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.005), 3, xyzs.__len__()*0.8, max_iterations) 
    a, b, c, d = m

    xyzMax = np.max(np.array(xyzs), axis=0)
    xyzMin = np.min(np.array(xyzs), axis=0)
    xyz = []
    ratio = 100
    for i in range(ratio):
        for j in range(ratio):
            x = xyzMin[0] + (xyzMax[0]-xyzMin[0]) * i / ratio
            z = xyzMin[2] + (xyzMax[2]-xyzMin[2]) * j / ratio
            y = (-d-a*x-c*z)/b
            xyz.append([x,y,z])

    meshData = MeshData()
    meshData.vert = np.array(xyz)
    write_obj('./data/temdata/results/plane1.obj', meshData)

    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

    plt.show()