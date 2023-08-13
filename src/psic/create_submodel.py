# -*- coding: utf-8 -*-
"""
生成规则网格
"""
import json
import os

import numpy as np
from numpy import ndarray, array, arange, meshgrid, concatenate

from psic.calc_fraction import calc_area_fraction, calc_volume_fraction
from psic.plot_model import plot_circle, plot_distribution, plot_sphere


def get_submodel(centers: ndarray, radiuses: ndarray, subsize: list) -> tuple[ndarray, ndarray]:
    """
    获取主模型在subsize区域内的子模型

    get_submodel(centers, radiuses, size)

    Parameters
    ----------
    centers : ndarray
        主模型的圆心数组
    radiuses : ndarray
        主模型的半径数组
    subsize : list
        子区域取值范围

    Returns
    -------
    sub_centers : ndarray
        子区域内的圆心坐标
    sub_radiuses : ndarray
        与sub_centers对应的半径
    """
    # 判断是否在区域内
    lower = (centers + radiuses) > array(subsize).T[0]
    upper = (centers - radiuses) < array(subsize).T[1]

    # 在子区域内部的布尔型矩阵
    is_in_cube_matrix = np.concatenate((lower, upper), axis=1)

    # 在子区域内部的布尔型列表
    is_in_cube = np.sum(is_in_cube_matrix, axis=1) >= 2 * len(subsize)

    sub_centers = centers[is_in_cube]
    sub_radiuses = radiuses[is_in_cube]

    return sub_centers, sub_radiuses


def create_submodel(model_file: str, model_id: int, size: list, ndiv: int, gap: float, out_path: str,
                    status: dict) -> int:
    """
    根据主模型生成ndiv等分后的子模型

    create_submodel(model_file, model_id, size, ndiv, gap, out_path, status)

    Parameters
    ----------
    model_file : str
        主模型文件路径
    model_id : int
        主模型编号
    size : list
        主模型区域取值范围
    ndiv : int
        在每个坐标轴方向等分数
    gap : float
        圆/球之间的间隔
    out_path : str
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
    geo_dimension = circles.shape[-1] - 1
    centers = circles[:, 0:geo_dimension]
    radiuses = circles[:, geo_dimension:geo_dimension + 1]
    box_dimension = len(size)
    if geo_dimension != box_dimension:
        raise NotImplementedError(
            f'the geometry dimension->{geo_dimension} is not equal to the dimension of given box->{box_dimension}')
    subsize = (array(size) / ndiv).tolist()
    a = [arange(s[0], s[1], s[1] / ndiv) for s in array(size)]
    b = meshgrid(*a)
    c = array([bb.flatten() for bb in b])

    for i in range(c.shape[1]):

        status['progress'] = int(i / c.shape[1] * 100)

        submodel_id = i + 1
        shift_centers = centers - c[:, i]
        sub_centers, sub_radiuses = get_submodel(shift_centers, radiuses, subsize)
        data = concatenate((sub_centers, sub_radiuses), axis=1)

        submodel_path = os.path.join(out_path, str(submodel_id))

        if not os.path.isdir(submodel_path):
            os.makedirs(submodel_path)

        filename = os.path.join(submodel_path, 'model.npy')
        np.save(filename, data)

        filename = os.path.join(submodel_path, 'model.png')
        if geo_dimension == 2:
            plot_circle(sub_centers, sub_radiuses - gap, subsize, filename, 150)
        elif geo_dimension >= 3:
            plot_sphere(sub_centers, sub_radiuses - gap, subsize, filename, (500, 500))
        else:
            raise NotImplementedError(f'the geometry dimension {geo_dimension} is not supported')

        filename = os.path.join(submodel_path, 'density.png')
        plot_distribution(sub_radiuses - gap, filename, 150)

        filename = os.path.join(submodel_path, 'model.msg')
        if geo_dimension == 2:
            fraction = calc_area_fraction(sub_centers, sub_radiuses - gap, subsize)
        elif geo_dimension >= 3:
            fraction = calc_volume_fraction(sub_centers, sub_radiuses - gap, subsize)
        else:
            raise NotImplementedError(f'the geometry dimension {geo_dimension} is not supported')
        message = {
            'model_id': model_id,
            'submodel_id': submodel_id,
            'fraction': fraction,
            'num_ball': len(sub_radiuses),
            'subsize': subsize,
            'location': (array(subsize) + c[:, i].reshape(-1, 1)).tolist(),
            'size': size,
            'gap': gap,
            'ndiv': ndiv
        }
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
    create_submodel('model.npy', 1, [[0, 1], [0, 1]], 4, 0.001, 'sub', status)
    # create_submodel('model.npy', 1, [[0, 1], [0, 1], [0, 1]], 2, 0.001, 'sub', status)
    print(status)
