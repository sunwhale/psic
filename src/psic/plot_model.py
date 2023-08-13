# -*- coding: utf-8 -*-
"""

"""
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.patches import Circle  # type: ignore
from numpy import ndarray

try:
    from mayavi import mlab  # type: ignore

    has_mayavi = True
except:
    has_mayavi = False
matplotlib.use('Agg')


def plot_circle(centers: ndarray, radiuses: ndarray, size: list, filename: str, dpi: int = 150) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i, _ in enumerate(centers):
        C = Circle(centers[i], radius=radiuses[i], facecolor=(1, 1, 1, 1), edgecolor=(0, 0, 0, 1), linewidth=None)
        ax.add_patch(C)
    ax.axis('equal')
    ax.set_box_aspect(1)
    ax.set_xlim(size[0])
    ax.set_ylim(size[1])
    ax.set_xlabel('x, mm')
    ax.set_ylabel('y, mm')
    fig.savefig(filename, dpi=dpi, transparent=False)
    plt.close(fig)


def plot_sphere(centers: ndarray, radiuses: ndarray, size: list, filename: str,
                dpi: tuple[int, int] = (500, 500)) -> None:
    x, y, z, r = centers[:, 0], centers[:, 1], centers[:, 2], radiuses[:, 0]
    if has_mayavi:
        mlab.options.offscreen = True
        fig = mlab.figure(size=(500, 500), bgcolor=(1, 1, 1))
        mlab.points3d(x, y, z, r * 2, scale_factor=1, resolution=30, mode="sphere")
        mlab.outline(fig)
        mlab.view(distance=4)
        mlab.savefig(filename, figure=fig, size=dpi)
        mlab.close(all=True)
    else:
        plot_circle(centers[:, 0:2], radiuses, size, filename)


def plot_distribution(radiuses: ndarray, filename: str, dpi: int = 150) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.hist(radiuses * 1000, bins=10, density=True)
    ax.set_xlabel('Radius, $\\mu$m')
    ax.set_ylabel('Density')
    fig.savefig(filename, dpi=dpi, transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    circles = np.load('model.npy')
    dim = circles.shape[-1] - 1
    centers = circles[:, 0:dim]
    radiuses = circles[:, dim:dim + 1]
    plot_circle(centers, radiuses, [[0, 1], [0, 1]], 'model.png', 300)
    # plot_sphere(centers, radiuses, [[0, 1], [0, 1], [0, 1]], 'model.png', (400, 400))
    plot_distribution(radiuses, 'density.png', 300)
