# -*- coding: utf-8 -*-
"""
生成规则网格
"""

import numpy as np
import matplotlib.pyplot as plt

# 本地文件
try:
    from calc_fraction import calc_area_fraction, calc_volume_fraction
    from plot_model import plot_circle, plot_sphere, plot_distribution
except:
    from psic.calc_fraction import calc_area_fraction, calc_volume_fraction
    from psic.plot_model import plot_circle, plot_sphere, plot_distribution


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


if __name__ == "__main__":

    circles = np.load('model.npy')
    dim = circles.shape[-1]-1
    centers = circles[:, 0:dim]
    radiuses = circles[:, dim:dim+1]

    size = [[0, 1], [0, 1], [0, 1]]
    sub_size = [[0, 0.1], [0, 0.1], [0, 0.1]]
    gap = 0.0015
    # plot_circle(circles, radiuses-gap, size, '1.png', 100)
    count = 0
    for dx in np.arange(0, 0.1, 0.1):
        for dy in np.arange(0, 0.1, 0.1):
            for dz in np.arange(0, 0.1, 0.1):
                count += 1
                # print(count)
                shift_centers = centers - np.array([dx, dy, dz])
                sub_centers, sub_radiuses = get_submodel(
                    shift_centers, radiuses, sub_size)
                plot_sphere(sub_centers, sub_radiuses-gap, sub_size, '1.png')
                print(count, calc_area_fraction(
                    sub_centers, sub_radiuses-gap, sub_size))
                name = ("%3s.npy" % count).replace(' ', '0')
                data = np.concatenate((sub_centers, sub_radiuses), axis=1)
                # np.save('.\\submodel\\' + model_name + '_' + name, data)
