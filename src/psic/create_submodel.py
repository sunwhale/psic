# -*- coding: utf-8 -*-
"""
生成规则网格
"""
import json
import os

import numpy as np

from psic.calc_fraction import calc_area_fraction, calc_volume_fraction
from psic.plot_model import plot_circle, plot_distribution, plot_sphere


def get_submodel(centers, radiuses, subsize):
    """
    获取主模型在subsize区域内的子模型

    get_submodel(centers, radiuses, size)

    Parameters
    ----------
    centers : array
        主模型的圆心数组
    radiuses : array
        主模型的半径数组
    subsize : list
        子区域取值范围

    Returns
    -------
    sub_centers : array
        子区域内的圆心坐标
    sub_radiuses : array
        与sub_centers对应的半径

    """
    # 判断是否在区域内
    lower = (centers + radiuses) > np.array(subsize).T[0]
    upper = (centers - radiuses) < np.array(subsize).T[1]

    # 在子区域内部的布尔型矩阵
    is_in_cube_matrix = np.concatenate((lower, upper), axis=1)

    # 在子区域内部的布尔型列表
    is_in_cube = np.sum(is_in_cube_matrix, axis=1) >= 2 * len(subsize)

    sub_centers = centers[is_in_cube]
    sub_radiuses = radiuses[is_in_cube]

    return sub_centers, sub_radiuses


def create_submodel(model_file, model_id, size, ndiv, gap, out_path, status):
    """
    根据主模型生成ndiv等分后的子模型

    create_submodel(model_file, model_id, size, ndiv, gap, out_path, status)

    Parameters
    ----------
    model_file : filepath
        主模型文件路径
    model_id : int
        主模型编号
    size : list
        主模型区域取值范围
    ndiv : int
        在每个坐标轴方向等分数
    gap : float
        圆/球之间的间隔
    out_path : path
        子文件输出根目录
    status : dict
        运行状态字典

    Process
    -------
    生成模型文件：model.npy
    生成参数文件：args.json
    生成信息文件：model.msg
    生成模型图片：model.png
    生成分布图片：density.png

    Returns
    -------
    0

    """

    status['status'] = 'Running'
    args = model_file, model_id, size, ndiv, gap, out_path, status

    circles = np.load(model_file)
    dim = circles.shape[-1] - 1
    centers = circles[:, 0:dim]
    radiuses = circles[:, dim:dim + 1]
    size = np.array(size)
    subsize = size / ndiv
    a = [np.arange(s[0], s[1], s[1] / ndiv) for s in size]
    b = np.meshgrid(*a)
    c = np.array([bb.flatten() for bb in b])

    for i in range(c.shape[1]):

        status['progress'] = int(i / c.shape[1] * 100)

        submodel_id = i + 1
        shift_centers = centers - c[:, i]
        sub_centers, sub_radiuses = get_submodel(shift_centers, radiuses, subsize)
        data = np.concatenate((sub_centers, sub_radiuses), axis=1)

        submodel_path = os.path.join(out_path, str(submodel_id))

        if not os.path.isdir(submodel_path):
            os.makedirs(submodel_path)

        filename = os.path.join(submodel_path, 'model.npy')
        np.save(filename, data)

        filename = os.path.join(submodel_path, 'model.png')
        if len(size) == 2:
            plot_circle(sub_centers, sub_radiuses - gap, subsize, filename, 150)
        if len(size) >= 3:
            plot_sphere(sub_centers, sub_radiuses - gap, subsize, filename, (500, 500))

        filename = os.path.join(submodel_path, 'density.png')
        plot_distribution(sub_radiuses - gap, filename, 150)

        filename = os.path.join(submodel_path, 'model.msg')
        message = {}
        if len(size) == 2:
            fraction = calc_area_fraction(sub_centers, sub_radiuses - gap, subsize)
        if len(size) >= 3:
            fraction = calc_volume_fraction(sub_centers, sub_radiuses - gap, subsize)
        message['model_id'] = model_id
        message['submodel_id'] = submodel_id
        message['fraction'] = fraction
        message['num_ball'] = len(sub_radiuses)
        message['subsize'] = subsize.tolist()
        message['location'] = (subsize + c[:, i].reshape(-1, 1)).tolist()
        message['size'] = size.tolist()
        message['gap'] = gap
        message['ndiv'] = ndiv
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(message, f, ensure_ascii=False)

        filename = os.path.join(submodel_path, 'args.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(args[:-1], f, ensure_ascii=False)

    status['progress'] = 100
    status['status'] = 'Done'

    return 0


if __name__ == "__main__":
    status = {'status': 'Submit', 'log': '', 'progress': 0}
    create_submodel('model.npy', 1, [[0, 1], [0, 1]], 4, 0, 'sub', status)
    # create_submodel('model3.npy', 1, [[0, 1], [0, 1], [0, 1]], 2, 0, 'sub', status)
    print(status)
