# -*- coding: utf-8 -*-
"""
生成四边形/六面体网格
"""
import json
import os

import numpy as np
from numpy import ndarray


def node_number(i: int | ndarray, j: int | ndarray, k: int | ndarray, l: int | ndarray, m: int | ndarray, n: int | ndarray) -> ndarray:
    """
    生成i，j，k对应的节点号

    Parameters
    ----------
    i : int | ndarray
        单元对应的1方向编号
    j : int | ndarray
        单元对应的2方向编号
    k : int | ndarray
        单元对应的3方向编号
    l : int | ndarray
        1方向的节点数
    m : int | ndarray
        2方向的节点数
    n : int | ndarray
        3方向的节点数

    Returns
    -------
    ndarray
        节点编号
    """
    return k * l * m + j * l + i  # type: ignore


def element_node_C3D8(i: ndarray, j: ndarray, k: ndarray, l: ndarray, m: ndarray, n: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    生成i，j，k对应的三维单元的节点号

    Parameters
    ----------
    i : ndarray
        单元对应的1方向编号
    j : ndarray
        单元对应的2方向编号
    k : ndarray
        单元对应的3方向编号
    l : ndarray
        1方向的节点数
    m : ndarray
        2方向的节点数
    n : ndarray
        3方向的节点数

    Returns
    -------
    n1 : ndarray
        单元的1号节点对应的总节点编号
    n2 : ndarray
        单元的2号节点对应的总节点编号
    n3 : ndarray
        单元的3号节点对应的总节点编号
    n4 : ndarray
        单元的4号节点对应的总节点编号
    n5 : ndarray
        单元的5号节点对应的总节点编号
    n6 : ndarray
        单元的6号节点对应的总节点编号
    n7 : ndarray
        单元的7号节点对应的总节点编号
    n8 : ndarray
        单元的8号节点对应的总节点编号
    """
    n1 = node_number(i, j, k, l, m, n)
    n2 = node_number(i + 1, j, k, l, m, n)
    n3 = node_number(i + 1, j + 1, k, l, m, n)
    n4 = node_number(i, j + 1, k, l, m, n)
    n5 = node_number(i, j, k + 1, l, m, n)
    n6 = node_number(i + 1, j, k + 1, l, m, n)
    n7 = node_number(i + 1, j + 1, k + 1, l, m, n)
    n8 = node_number(i, j + 1, k + 1, l, m, n)
    return n1, n2, n3, n4, n5, n6, n7, n8


def element_node_CPE4(i: ndarray, j: ndarray, l: ndarray, m: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    生成i，j对应的二维单元的节点号

    Parameters
    ----------
    i : ndarray
        单元对应的1方向编号
    j : ndarray
        单元对应的2方向编号
    l : ndarray
        1方向的节点数
    m : ndarray
        2方向的节点数

    Returns
    -------
    n1 : ndarray
        单元的1号节点对应的总节点编号
    n2 : ndarray
        单元的2号节点对应的总节点编号
    n3 : ndarray
        单元的3号节点对应的总节点编号
    n4 : ndarray
        单元的4号节点对应的总节点编号
    """
    k = 0
    n = 0
    n1 = node_number(i, j, k, l, m, n)
    n2 = node_number(i + 1, j, k, l, m, n)
    n3 = node_number(i + 1, j + 1, k, l, m, n)
    n4 = node_number(i, j + 1, k, l, m, n)
    return n1, n2, n3, n4


def create_elements_centroids(node_shape: list, dimension: list) -> tuple[ndarray, ndarray] | tuple[
    ndarray, ndarray, ndarray]:
    """
    生成全体单元中心点的坐标数组

    Parameters
    ----------
    node_shape : list
        节点在每个坐标轴方向的数量列表，例如[65, 65]代表x和y方向分别设置65个等分点
    dimension : list
        相对于原点，RVE的尺寸大小

    Returns
    -------
    x, y, z
        全体单元中心点在不同方向上的坐标数组
    """
    if len(node_shape) == 2:
        x = np.linspace(dimension[0] / (node_shape[0] - 1) / 2, dimension[0] -
                        dimension[0] / (node_shape[0] - 1) / 2, node_shape[0] - 1)
        y = np.linspace(dimension[1] / (node_shape[1] - 1) / 2, dimension[1] -
                        dimension[1] / (node_shape[1] - 1) / 2, node_shape[1] - 1)
        x, y = np.meshgrid(x, y)
        return x, y
    elif len(node_shape) == 3:
        x = np.linspace(dimension[0] / (node_shape[0] - 1) / 2, dimension[0] -
                        dimension[0] / (node_shape[0] - 1) / 2, node_shape[0] - 1)
        y = np.linspace(dimension[1] / (node_shape[1] - 1) / 2, dimension[1] -
                        dimension[1] / (node_shape[1] - 1) / 2, node_shape[1] - 1)
        z = np.linspace(dimension[2] / (node_shape[2] - 1) / 2, dimension[2] -
                        dimension[2] / (node_shape[2] - 1) / 2, node_shape[2] - 1)
        x, y, z = np.meshgrid(x, y, z)
        return x, y, z
    else:
        raise NotImplementedError(f'the geometry dimension {len(node_shape)} is not supported')


def create_nodes_coordinates(node_shape: list, dimension: list) -> tuple[ndarray, ndarray] | tuple[
    ndarray, ndarray, ndarray]:
    """
    生成全体节点坐标数组

    Parameters
    ----------
    node_shape : list
        节点在每个坐标轴方向的数量列表，例如[65, 65]代表x和y方向分别设置65个等分点
    dimension : list
        相对于原点，RVE的尺寸大小

    Returns
    -------
    x, y, z
        全体节点在不同方向上的坐标数组
    """
    if len(node_shape) == 2:
        x = np.linspace(0, dimension[0], node_shape[0])
        y = np.linspace(0, dimension[1], node_shape[1])
        x, y = np.meshgrid(x, y)
        return x, y
    elif len(node_shape) == 3:
        x = np.linspace(0, dimension[0], node_shape[0])
        y = np.linspace(0, dimension[1], node_shape[1])
        z = np.linspace(0, dimension[2], node_shape[2])
        y, z, x = np.meshgrid(x, y, z)
        return x, y, z
    else:
        raise NotImplementedError(f'the geometry dimension {len(node_shape)} is not supported')


def create_nodes_indices(node_shape: list) -> ndarray:
    """
    生成全体节点索引编号

    Parameters
    ----------
    node_shape : list
        节点在每个坐标轴方向的数量列表，例如[65, 65]代表x和y方向分别设置65个等分点

    Returns
    -------
    ndarray
        与节点位置对应的索引编号数组，[i, j, k]->节点全局索引编号
    """
    node_size = 1
    for n in node_shape:
        node_size *= n
    return np.arange(node_size).reshape(node_shape)


def create_elements_indices(node_shape: list) -> ndarray:
    """
    生成全体单元索引编号

    Parameters
    ----------
    node_shape : list
        节点在每个坐标轴方向的数量列表，例如[65, 65]代表x和y方向分别设置65个等分点

    Returns
    -------
    ndarray
        与单元位置对应的索引编号数组，[i, j, k]->单元全局索引编号
    """
    element_shape = [n - 1 for n in node_shape]
    element_size = 1
    for n in node_shape:
        element_size *= (n - 1)
    return np.arange(element_size).reshape(element_shape)


def create_lmn(node_shape: list) -> tuple[ndarray, ndarray] | tuple[ndarray, ndarray, ndarray]:
    """
    生成l, m, n

    Parameters
    ----------
    node_shape : list
        节点在每个坐标轴方向的数量列表，例如[65, 65]代表x和y方向分别设置65个等分点

    Returns
    -------
    m, l, n
        每个坐标方向上的节点数
    """
    if len(node_shape) == 2:
        m = node_shape[0]
        l = node_shape[1]
        return l, m
    elif len(node_shape) == 3:
        m = node_shape[0]
        l = node_shape[1]
        n = node_shape[2]
        return l, m, n
    else:
        raise NotImplementedError(f'the geometry dimension {len(node_shape)} is not supported')
        # return 0


def create_ijk(node_shape: list) -> tuple[ndarray, ndarray] | tuple[ndarray, ndarray, ndarray]:
    """
    生成i, j, k

    Parameters
    ----------
    node_shape : list
        节点在每个坐标轴方向的数量列表，例如[65, 65]代表x和y方向分别设置65个等分点

    Returns
    -------
    j, k, i
        每个坐标方向上的单元编号
    """
    element_shape = [n - 1 for n in node_shape]
    ijk = [np.arange(n) for n in element_shape]
    if len(node_shape) == 2:
        i, j = np.meshgrid(ijk[0], ijk[1])
        return i, j
    elif len(node_shape) == 3:
        j, k, i = np.meshgrid(ijk[0], ijk[1], ijk[2])
        return i, j, k
    else:
        raise NotImplementedError(f'the geometry dimension {len(node_shape)} is not supported')
        # return 0


def write_input_file(dim: int, nodes: list[ndarray], elements: list[ndarray], element_type: str, element_sets_dict: dict,
                     output_path: str) -> None:
    filename = os.path.join(output_path, 'Model-1.inp')
    outfile = open(filename, 'w')

    outfile.write('*Part, name=PART-1\n')
    outfile.write('*Node\n')

    if dim == 2:
        for num, x, y in np.nditer(nodes):
            outfile.write('%s, %s, %s, %s\n' % (num + 1, x, y, 0))

        outfile.write('*Element, type=%s\n' % element_type)
        for num, n1, n2, n3, n4 in np.nditer(elements):
            outfile.write('%s, %s, %s, %s, %s\n' %
                          (num + 1, n1 + 1, n2 + 1, n3 + 1, n4 + 1))

    elif dim >= 3:
        for num, x, y, z in np.nditer(nodes):
            outfile.write('%s, %s, %s, %s,%s\n' % (num + 1, x, y, z, 0))

        outfile.write('*Element, type=%s\n' % element_type)
        for num, n1, n2, n3, n4, n5, n6, n7, n8 in np.nditer(elements):
            outfile.write('%s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
                num + 1, n1 + 1, n2 + 1, n3 + 1, n4 + 1, n5 + 1, n6 + 1, n7 + 1, n8 + 1))

    for key in element_sets_dict.keys():
        outfile.write('*Elset, elset=ELSET_%s\n' % key)
        count = 0
        for i in element_sets_dict[key]:
            count += 1
            if count == len(element_sets_dict[key]):
                end = '\n'
            elif count % 8 == 0:
                end = '\n'
            else:
                end = ','
            outfile.write(str(i) + end)

    outfile.write('*Nset, nset=NSET_ALL\n')
    count = 0
    for node in np.nditer(nodes[0]):
        count += 1
        if count == len(nodes[0]):
            end = '\n'
        elif count % 8 == 0:
            end = '\n'
        else:
            end = ','
        outfile.write(str(node + 1) + end)  # type: ignore

    if dim == 2:
        x0 = min(nodes[1])
        x1 = max(nodes[1])
        y0 = min(nodes[2])
        y1 = max(nodes[2])
        node_sets_dict = {
            'X0': nodes[0][nodes[1] == x0],
            'X1': nodes[0][nodes[1] == x1],
            'Y0': nodes[0][nodes[2] == y0],
            'Y1': nodes[0][nodes[2] == y1],
            'X0Y0': nodes[0][(nodes[1] == x0) & (nodes[2] == y0)],
            'X0Y1': nodes[0][(nodes[1] == x0) & (nodes[2] == y1)],
            'X1Y0': nodes[0][(nodes[1] == x1) & (nodes[2] == y0)],
            'X1Y1': nodes[0][(nodes[1] == x1) & (nodes[2] == y1)],
            'X0-X0Y0': nodes[0][(nodes[1] == x0) & (~((nodes[1] == x0) & (nodes[2] == y0)))],
            'X1-X1Y0': nodes[0][(nodes[1] == x1) & (~((nodes[1] == x1) & (nodes[2] == y0)))]
        }

    elif dim == 3:
        x0 = min(nodes[1])
        x1 = max(nodes[1])
        y0 = min(nodes[2])
        y1 = max(nodes[2])
        z0 = min(nodes[3])
        z1 = max(nodes[3])
        node_sets_dict = {
            'X0': nodes[0][nodes[1] == x0],
            'X1': nodes[0][nodes[1] == x1],
            'Y0': nodes[0][nodes[2] == y0],
            'Y1': nodes[0][nodes[2] == y1],
            'Z0': nodes[0][nodes[3] == z0],
            'Z1': nodes[0][nodes[3] == z1],
            'X0Y0Z0': nodes[0][(nodes[1] == x0) & (nodes[2] == y0) & (nodes[3] == z0)],
            'X0Y0Z1': nodes[0][(nodes[1] == x0) & (nodes[2] == y0) & (nodes[3] == z1)],
            'X0Y1Z0': nodes[0][(nodes[1] == x0) & (nodes[2] == y1) & (nodes[3] == z0)],
            'X0Y1Z1': nodes[0][(nodes[1] == x0) & (nodes[2] == y1) & (nodes[3] == z1)],
            'X1Y0Z0': nodes[0][(nodes[1] == x1) & (nodes[2] == y0) & (nodes[3] == z0)],
            'X1Y0Z1': nodes[0][(nodes[1] == x1) & (nodes[2] == y0) & (nodes[3] == z1)],
            'X1Y1Z0': nodes[0][(nodes[1] == x1) & (nodes[2] == y1) & (nodes[3] == z0)],
            'X1Y1Z1': nodes[0][(nodes[1] == x1) & (nodes[2] == y1) & (nodes[3] == z1)]
        }

    else:
        raise NotImplementedError(f'the geometry dimension {dim} is not supported')

    for key in node_sets_dict.keys():
        outfile.write('*Nset, nset=%s\n' % key)
        count = 0
        for node in np.nditer(node_sets_dict[key]):
            count += 1
            if count == len(node_sets_dict[key]):
                end = '\n'
            elif count % 8 == 0:
                end = '\n'
            else:
                end = ','
            outfile.write(str(node + 1) + end)  # type: ignore

    outfile.close()


def mesh(gap: float, dimension: list, node_shape: list, element_type: str, model_path: str, output_path: str,
         status: dict, is_interface: bool = True) -> None:
    model_file = os.path.join(model_path, 'model.npy')
    circles = np.load(model_file)
    dim = circles.shape[-1] - 1
    if dim == 2:
        node_index = create_nodes_indices(node_shape)
        element_index = create_elements_indices(node_shape)

        x, y = create_nodes_coordinates(node_shape, dimension)  # type: ignore
        ecx, ecy = create_elements_centroids(node_shape, dimension)  # type: ignore

        l, m = create_lmn(node_shape)  # type: ignore
        i, j = create_ijk(node_shape)  # type: ignore
        n1, n2, n3, n4 = element_node_CPE4(i, j, l, m)

        node_index = node_index.flatten()
        x = x.flatten()
        y = y.flatten()
        nodes = [node_index, x, y]

        element_index = element_index.flatten()
        ecx = ecx.flatten()
        ecy = ecy.flatten()
        n1 = n1.flatten()
        n2 = n2.flatten()
        n3 = n3.flatten()
        n4 = n4.flatten()
        elements = [element_index, n1, n2, n3, n4]

        element_sets = np.zeros(len(element_index))

        n1x = x[n1]
        n1y = y[n1]
        n2x = x[n2]
        n2y = y[n2]
        n3x = x[n3]
        n3y = y[n3]
        n4x = x[n4]
        n4y = y[n4]

        if not is_interface:
            for i, circle in enumerate(circles):
                cx = circle[0]
                cy = circle[1]
                r = circle[2] - gap
                index = (ecx - cx) ** 2 + (ecy - cy) ** 2 < r ** 2
                element_sets[index] = int(i) + 1
        else:
            for i, circle in enumerate(circles):
                cx = circle[0]
                cy = circle[1]
                r = circle[2] - gap
                index1 = (n1x - cx) ** 2 + (n1y - cy) ** 2 < r ** 2
                index2 = (n2x - cx) ** 2 + (n2y - cy) ** 2 < r ** 2
                index3 = (n3x - cx) ** 2 + (n3y - cy) ** 2 < r ** 2
                index4 = (n4x - cx) ** 2 + (n4y - cy) ** 2 < r ** 2
                index_inner = index1 & index2 & index3 & index4
                element_sets[index_inner] = int(i) + 1
                index_inter = (index1 | index2 | index3 | index4) ^ index_inner
                element_sets[index_inter] = -1

    elif dim >= 3:
        node_index = create_nodes_indices(node_shape)
        element_index = create_elements_indices(node_shape)

        x, y, z = create_nodes_coordinates(node_shape, dimension)  # type: ignore
        ecx, ecy, ecz = create_elements_centroids(node_shape, dimension)  # type: ignore

        l, m, n = create_lmn(node_shape)  # type: ignore
        i, j, k = create_ijk(node_shape)  # type: ignore
        n1, n2, n3, n4, n5, n6, n7, n8 = element_node_C3D8(i, j, k, l, m, n)

        node_index = node_index.flatten()
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        nodes = [node_index, x, y, z]

        element_index = element_index.flatten()
        ecx = ecx.flatten()
        ecy = ecy.flatten()
        ecz = ecz.flatten()
        n1 = n1.flatten()
        n2 = n2.flatten()
        n3 = n3.flatten()
        n4 = n4.flatten()
        n5 = n5.flatten()
        n6 = n6.flatten()
        n7 = n7.flatten()
        n8 = n8.flatten()
        elements = [element_index, n1, n2, n3, n4, n5, n6, n7, n8]
        element_sets = np.zeros(len(element_index))

        n1x = x[n1]
        n1y = y[n1]
        n1z = z[n1]
        n2x = x[n2]
        n2y = y[n2]
        n2z = z[n2]
        n3x = x[n3]
        n3y = y[n3]
        n3z = z[n3]
        n4x = x[n4]
        n4y = y[n4]
        n4z = z[n4]
        n5x = x[n5]
        n5y = y[n5]
        n5z = z[n5]
        n6x = x[n6]
        n6y = y[n6]
        n6z = z[n6]
        n7x = x[n7]
        n7y = y[n7]
        n7z = z[n7]
        n8x = x[n8]
        n8y = y[n8]
        n8z = z[n8]

        if not is_interface:
            for i, circle in enumerate(circles):
                cx = circle[0]
                cy = circle[1]
                cz = circle[2]
                r = circle[-1] - gap
                index = (ecx - cx) ** 2 + (ecy - cy) ** 2 + (ecz - cz) ** 2 < r ** 2
                element_sets[index] = int(i) + 1
        else:
            for i, circle in enumerate(circles):
                cx = circle[0]
                cy = circle[1]
                cz = circle[2]
                r = circle[-1] - gap
                index1 = (n1x - cx) ** 2 + (n1y - cy) ** 2 + (n1z - cz) ** 2 < r ** 2
                index2 = (n2x - cx) ** 2 + (n2y - cy) ** 2 + (n2z - cz) ** 2 < r ** 2
                index3 = (n3x - cx) ** 2 + (n3y - cy) ** 2 + (n3z - cz) ** 2 < r ** 2
                index4 = (n4x - cx) ** 2 + (n4y - cy) ** 2 + (n4z - cz) ** 2 < r ** 2
                index5 = (n5x - cx) ** 2 + (n5y - cy) ** 2 + (n5z - cz) ** 2 < r ** 2
                index6 = (n6x - cx) ** 2 + (n6y - cy) ** 2 + (n6z - cz) ** 2 < r ** 2
                index7 = (n7x - cx) ** 2 + (n7y - cy) ** 2 + (n7z - cz) ** 2 < r ** 2
                index8 = (n8x - cx) ** 2 + (n8y - cy) ** 2 + (n8z - cz) ** 2 < r ** 2
                index_inner = index1 & index2 & index3 & index4 & index5 & index6 & index7 & index8
                element_sets[index_inner] = int(i) + 1
                index_inter = (index1 | index2 | index3 | index4 | index5 | index6 | index7 | index8) ^ index_inner
                element_sets[index_inter] = -1

    else:
        raise NotImplementedError(f'the geometry dimension {dim} is not supported')

    element_sets_names = np.unique(element_sets)

    element_sets_dict: dict = {}
    for name in element_sets_names:
        element_sets_dict[int(name)] = []
        element_sets_dict['MATRIX'] = []
        element_sets_dict['PARTICLES'] = []
        element_sets_dict['INTERFACES'] = []
        element_sets_dict['ALL'] = []
    for i, _ in enumerate(element_sets):
        element_sets_dict[int(element_sets[i])].append(element_index[i] + 1)
        if int(element_sets[i]) == 0:
            element_sets_dict['MATRIX'].append(element_index[i] + 1)
        elif int(element_sets[i]) == -1:
            element_sets_dict['INTERFACES'].append(element_index[i] + 1)
        else:
            element_sets_dict['PARTICLES'].append(element_index[i] + 1)
        element_sets_dict['ALL'].append(element_index[i] + 1)

    write_input_file(dim, nodes, elements, element_type, element_sets_dict, output_path)

    status['message']['element_sets_names'] = element_sets_names.tolist()
    status['message']['nodes_number'] = len(node_index)
    status['message']['elements_number'] = len(element_index)


def create_mesh(gap: float, size: list, dimension: list, node_shape: list, element_type: str, model_path: str, output_path: str, status: dict) -> int:
    args = gap, size, dimension, node_shape, element_type, model_path, output_path, status

    status['message'] = {}

    mesh(gap, dimension, node_shape, element_type, model_path, output_path, status)

    filename = os.path.join(output_path, 'model.msg')
    status['message']['node_shape'] = node_shape
    status['message']['dimension'] = dimension
    status['message']['size'] = size
    status['message']['gap'] = gap
    status['message']['element_type'] = element_type
    status['message']['gap'] = gap
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(status['message'], f, ensure_ascii=False)

    filename = os.path.join(output_path, 'args.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(args[:-1], f, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    # gap = 0.0
    # size = [[0.0, 0.125], [0.0, 0.125], [0.0, 0.125]]
    # dimension = [s[1] for s in size]
    # node_shape = [65, 65, 65]
    # element_type = 'C3D8T'
    # model_path = ''
    # output_path = ''
    # status = {'status': 'Submit', 'log': '', 'progress': 0}
    # args = (gap, size, dimension, node_shape, element_type, model_path, output_path, status)
    # print(status)
    # create_mesh(*args)
    # print(status)

    gap = 0.001
    size = [[0.0, 1.0], [0.0, 1.0]]
    dimension = [s[1] for s in size]
    node_shape = [65, 65]
    element_type = 'CPE4'
    model_path = ''
    output_path = ''
    status = {'status': 'Submit', 'log': '', 'progress': 0}
    args = (gap, size, dimension, node_shape, element_type, model_path, output_path, status)
    print(status)
    create_mesh(*args)
    print(status)
