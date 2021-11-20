import numpy as np
from ransac import *

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[:10, :10]
        return xx, yy, (-d - a * xx - b * yy) / c

    n = 103200
    max_iterations = 100
    goal_inliers = n * 0.5

    # test data
    # xyzs = np.random.random((n, 3)) * 10
    # xyzs[:50, 2:] = xyzs[:50, :1]

    f = open('joints.obj')

    def read_point(file_name):
        verts = []
        f = open(file_name)
        line = f.readline()
        while line:
            line_data = line.split()
            verts.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
            #color.append([float(line_data[4]),float(line_data[5]),float(line_data[6])])
            line = f.readline()
        f.close()
        return verts#,color
    xyzs = np.array(read_point('joints.obj'))


    ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

    # RANSAC
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

    plt.show()
