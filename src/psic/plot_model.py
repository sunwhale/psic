# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mayavi import mlab


def plot_circle(centers, radiuses, size, filename, dpi):
    fig, ax = plt.subplots(1, 1)
    for i in range(len(centers)):
        C = Circle(centers[i], radius=radiuses[i], facecolor=(1, 1, 1, 1), edgecolor=(0, 0, 0, 1), linewidth=None)
        ax.add_patch(C)
    ax.axis('equal')
    ax.set_box_aspect(1)
    ax.set_xlim(size[0])
    ax.set_ylim(size[1])
    fig.savefig(filename, dpi=dpi, transparent=True)


def plot_sphere(centers, radiuses, size, filename, dpi):
    x, y, z, r = centers[:, 0], centers[:, 1], centers[:, 2], radiuses[:, 0]
    fig = mlab.points3d(x, y, z, r*2, scale_factor=1, resolution=30, mode="sphere")
    mlab.outline(fig)
    mlab.view(distance=4)
    mlab.savefig(filename, figure=fig, size=dpi)
    fig.module_manager.source.save_output('model.vtk')
    mlab.close(all=True)
    

if __name__ == "__main__":
    circles = np.load('model.npy')
    dim = circles.shape[-1]-1
    centers = circles[:, 0:dim]
    radiuses = circles[:, dim:dim+1]
    plot_circle(centers, radiuses, [[0,1], [0,1]], 'model.png', 300)
    plot_sphere(centers, radiuses, [[0, 1], [0, 1], [0, 1]], 'model.png', (400,400))

