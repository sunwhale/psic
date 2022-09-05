# -*- coding: utf-8 -*-
"""
生成规则网格
"""

import numpy as np

# 本地文件
try:
    from calc_fraction import calc_area_fraction, calc_volume_fraction
    from plot_model import plot_circle, plot_distribution, plot_sphere
except ImportError:
    from psic.calc_fraction import calc_area_fraction, calc_volume_fraction
    from psic.plot_model import plot_circle, plot_distribution, plot_sphere


def get_submodel(centers, radiuses, size):
    # 判断是否在区域内
    lower = (centers + radiuses) > np.array(size).T[0]
    upper = (centers - radiuses) < np.array(size).T[1]

    # 在子区域内部的布尔型矩阵
    is_in_cube_matrix = np.concatenate((lower, upper), axis=1)

    # 在子区域内部的布尔型列表，四条边都在内部则为True
    is_in_cube = np.sum(is_in_cube_matrix, axis=1) >= 2*len(size)

    sub_centers = centers[is_in_cube]
    sub_radiuses = radiuses[is_in_cube]

    return sub_centers, sub_radiuses


def create_submodel(model, size, div):
    circles = np.load(model)
    dim = circles.shape[-1]-1
    centers = circles[:, 0:dim]
    radiuses = circles[:, dim:dim+1]
    gap = 0.0015
    size = np.array(size)
    sub_size = size/div
    a = [np.arange(s[0], s[1], s[1]/div) for s in size]
    b = np.meshgrid(*a)
    c = np.array([bb.flatten() for bb in b])

    # 如果不等分
    # div = [3, 4, 5]
    # sub_size = size/np.array(div).reshape(-1, 1)
    # a = [np.arange(s[0], s[1], s[1]/div[i]) for i, s in enumerate(size)]

    for i in range(c.shape[1]):
        shift_centers = centers - c[:, i]
        sub_centers, sub_radiuses = get_submodel(shift_centers, radiuses, sub_size)
        # plot_circle(sub_centers, sub_radiuses-gap, sub_size, '1.png')
        print(i+1, calc_area_fraction(sub_centers, sub_radiuses-gap, sub_size))
        data = np.concatenate((sub_centers, sub_radiuses), axis=1)
        # np.save('.\\submodel\\' + model_name + '_' + name, data)


if __name__ == "__main__":
    # create_submodel('model3.npy', [[0, 1], [0, 1], [0, 1]], 2)
    create_submodel('model.npy', [[0, 1], [0, 1]], 4)
