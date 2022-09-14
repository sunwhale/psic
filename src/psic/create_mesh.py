# -*- coding: utf-8 -*-
"""
生成四边形/六面体网格
"""

import numpy as np
import os
import json


def node_number(i, j, k, l, m, n):
    '''
    生成i，j，k对应的节点号

    Parameters
    ----------
    i : int
        单元对应的1方向编号
    j : int
        单元对应的2方向编号
    k : int
        单元对应的3方向编号
    l : int
        1方向的节点数
    m : int
        2方向的节点数
    n : int
        3方向的节点数

    Returns
    -------
    int
        节点编号

    '''
    return k*l*m + j*l + i


def element_node_C3D8(i, j, k, l, m, n):
    '''
    生成i，j，k对应的三维单元的节点号

    Parameters
    ----------
    i : int
        单元对应的1方向编号
    j : int
        单元对应的2方向编号
    k : int
        单元对应的3方向编号
    l : int
        1方向的节点数
    m : int
        2方向的节点数
    n : int
        3方向的节点数

    Returns
    -------
    n1 : int
        单元的1节点对应的总节点编号
    n2 : int
        单元的2节点对应的总节点编号
    n3 : int
        单元的3节点对应的总节点编号
    n4 : int
        单元的4节点对应的总节点编号
    n5 : int
        单元的5节点对应的总节点编号
    n6 : int
        单元的6节点对应的总节点编号
    n7 : int
        单元的7节点对应的总节点编号
    n8 : int
        单元的8节点对应的总节点编号

    '''
    n1 = node_number(i, j, k, l, m, n)
    n2 = node_number(i+1, j, k, l, m, n)
    n3 = node_number(i+1, j+1, k, l, m, n)
    n4 = node_number(i, j+1, k, l, m, n)
    n5 = node_number(i, j, k+1, l, m, n)
    n6 = node_number(i+1, j, k+1, l, m, n)
    n7 = node_number(i+1, j+1, k+1, l, m, n)
    n8 = node_number(i, j+1, k+1, l, m, n)
    return n1, n2, n3, n4, n5, n6, n7, n8


def element_node_CPE4(i, j, l, m):
    '''
    生成i，j对应的二维单元的节点号

    Parameters
    ----------
    i : int
        单元对应的1方向编号
    j : int
        单元对应的2方向编号
    l : int
        1方向的节点数
    m : int
        2方向的节点数

    Returns
    -------
    n1 : array
        单元的1节点对应的总节点编号
    n2 : array
        单元的2节点对应的总节点编号
    n3 : array
        单元的3节点对应的总节点编号
    n4 : array
        单元的4节点对应的总节点编号

    '''
    k = 0
    n = 0
    n1 = node_number(i, j, k, l, m, n)
    n2 = node_number(i+1, j, k, l, m, n)
    n3 = node_number(i+1, j+1, k, l, m, n)
    n4 = node_number(i, j+1, k, l, m, n)
    return n1, n2, n3, n4


def element_centroid(node_shape, dimension):
    '''
    生成单元中心点的坐标数组

    Parameters
    ----------
    node_shape : list
        节点矩阵维度参数
    dimension : list
        RVE的尺寸大小

    Returns
    -------
    array
        单元中心点在不同方向上的坐标数组

    '''
    if len(node_shape) == 2:
        x = np.linspace(dimension[0]/(node_shape[0]-1)/2, dimension[0] -
                        dimension[0]/(node_shape[0]-1)/2, node_shape[0]-1)
        y = np.linspace(dimension[1]/(node_shape[1]-1)/2, dimension[1] -
                        dimension[1]/(node_shape[1]-1)/2, node_shape[1]-1)
        x, y = np.meshgrid(x, y)
        return x, y
    if len(node_shape) == 3:
        x = np.linspace(dimension[0]/(node_shape[0]-1)/2, dimension[0] -
                        dimension[0]/(node_shape[0]-1)/2, node_shape[0]-1)
        y = np.linspace(dimension[1]/(node_shape[1]-1)/2, dimension[1] -
                        dimension[1]/(node_shape[1]-1)/2, node_shape[1]-1)
        z = np.linspace(dimension[2]/(node_shape[2]-1)/2, dimension[2] -
                        dimension[2]/(node_shape[2]-1)/2, node_shape[2]-1)
        x, y, z = np.meshgrid(x, y, z)
        return x, y, z
    return 0


def create_node_coordinate(node_shape, dimension):
    '''
    生成节点坐标数组

    Parameters
    ----------
    node_shape : list
        节点矩阵维度参数
    dimension : list
        RVE的尺寸大小

    Returns
    -------
    array
        节点在不同方向上的坐标数组

    '''
    if len(node_shape) == 2:
        x = np.linspace(0, dimension[0], node_shape[0])
        y = np.linspace(0, dimension[1], node_shape[1])
        x, y = np.meshgrid(x, y)
        return x, y
    if len(node_shape) == 3:
        x = np.linspace(0, dimension[0], node_shape[0])
        y = np.linspace(0, dimension[1], node_shape[1])
        z = np.linspace(0, dimension[2], node_shape[2])
        y, z, x = np.meshgrid(x, y, z)
        return x, y, z


def create_node_index(node_shape):
    '''
    生成节点索引编号

    Parameters
    ----------
    node_shape : list
        节点形状

    Returns
    -------
    array
        与节点位置对应的索引编号数组

    '''
    node_size = 1
    for n in node_shape:
        node_size *= n
    return np.arange(node_size).reshape(node_shape)


def create_element_index(node_shape):
    '''
    生成单元索引编号

    Parameters
    ----------
    node_shape : list
        节点形状

    Returns
    -------
    array
        与单元位置对应的索引编号数组

    '''
    element_shape = [n-1 for n in node_shape]
    element_size = 1
    for n in node_shape:
        element_size *= (n-1)
    return np.arange(element_size).reshape(element_shape)


def create_lmn(node_shape):
    '''
    生成l, m, n

    Parameters
    ----------
    node_shape : list
        节点形状

    Returns
    -------
    tuple
        每个坐标方向上的节点数

    '''
    if len(node_shape) == 2:
        m = node_shape[0]
        l = node_shape[1]
        return l, m
    elif len(node_shape) == 3:
        m = node_shape[0]
        l = node_shape[1]
        n = node_shape[2]
        return l, m, n
    return 0


def create_ijk(node_shape):
    '''
    生成i, j, k

    Parameters
    ----------
    node_shape : list
        节点形状

    Returns
    -------
    tuple
        每个坐标方向上的单元编号

    '''
    element_shape = [n-1 for n in node_shape]
    ijk = [np.arange(n) for n in element_shape]
    if len(node_shape) == 2:
        i, j = np.meshgrid(ijk[0], ijk[1])
        return i, j
    elif len(node_shape) == 3:
        j, k, i = np.meshgrid(ijk[0], ijk[1], ijk[2])
        return i, j, k
    return 0


def write_input_file(dim, nodes, elements, element_type, element_sets_dict, output_path):
    filename = os.path.join(output_path, 'Model-1.inp')
    outfile = open(filename, 'w')

    outfile.write('*Part, name=PART-1\n')
    outfile.write('*Node\n')

    if dim == 2:
        for num, x, y in np.nditer(nodes):
            outfile.write('%s, %s, %s, %s\n' % (num+1, x, y, 0))

        outfile.write('*Element, type=%s\n' % element_type)
        for num, n1, n2, n3, n4 in np.nditer(elements):
            outfile.write('%s, %s, %s, %s, %s\n' %
                          (num+1, n1+1, n2+1, n3+1, n4+1))

    elif dim >= 3:
        for num, x, y, z in np.nditer(nodes):
            outfile.write('%s, %s, %s, %s,%s\n' % (num+1, x, y, z, 0))

        outfile.write('*Element, type=%s\n' % element_type)
        for num, n1, n2, n3, n4, n5, n6, n7, n8 in np.nditer(elements):
            outfile.write('%s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
                num+1, n1+1, n2+1, n3+1, n4+1, n5+1, n6+1, n7+1, n8+1))

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
        outfile.write(str(node+1) + end)

    outfile.write('*End Part\n')
    outfile.close()


def mesh(gap, dimension, node_shape, element_type, model_path, output_path, status):
    model_file = os.path.join(model_path, 'model.npy')
    circles = np.load(model_file)
    dim = circles.shape[-1]-1
    if dim == 2:
        node_index = create_node_index(node_shape)
        element_index = create_element_index(node_shape)

        x, y = create_node_coordinate(node_shape, dimension)
        ecx, ecy = element_centroid(node_shape, dimension)

        l, m = create_lmn(node_shape)
        i, j = create_ijk(node_shape)
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

        for i, circle in enumerate(circles):
            cx = circle[0]
            cy = circle[1]
            r = circle[2] - gap
            index = (ecx - cx)**2 + (ecy - cy)**2 < r**2
            element_sets[index] = int(i) + 1

    elif dim >= 3:
        node_index = create_node_index(node_shape)
        element_index = create_element_index(node_shape)
        x, y, z = create_node_coordinate(node_shape, dimension)
        ecx, ecy, ecz = element_centroid(node_shape, dimension)

        l, m, n = create_lmn(node_shape)
        i, j, k = create_ijk(node_shape)
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

        for i, circle in enumerate(circles):
            cx = circle[0]
            cy = circle[1]
            cz = circle[2]
            r = circle[-1] - gap
            index = (ecx - cx)**2 + (ecy - cy)**2 + (ecz-cz)**2 < r**2
            element_sets[index] = int(i) + 1

    element_sets_names = np.unique(element_sets)
    
    element_sets_dict = {}
    for name in element_sets_names:
        element_sets_dict[int(name)] = []
        element_sets_dict['MATRIX'] = []
        element_sets_dict['PARTICLES'] = []
        element_sets_dict['ALL'] = []
    for i in range(len(element_sets)):
        element_sets_dict[int(element_sets[i])].append(element_index[i]+1)
        if int(element_sets[i]) == 0:
            element_sets_dict['MATRIX'].append(element_index[i]+1)
        else:
            element_sets_dict['PARTICLES'].append(element_index[i]+1)
        element_sets_dict['ALL'].append(element_index[i]+1)

    write_input_file(dim, nodes, elements, element_type, element_sets_dict, output_path)
    
    status['message']['element_sets_names'] = str(element_sets_names)
    status['message']['nodes_number'] = len(node_index)
    status['message']['elements_number'] = len(element_index)


def create_mesh(gap, size, dimension, node_shape, element_type, model_path, output_path, status):

    gap, size, dimension, node_shape, element_type, model_path, output_path, status = args

    status['status'] = 'Running'
    status['message'] = {}

    mesh(gap, dimension, node_shape, element_type, model_path, output_path, status)

    filename = os.path.join(model_path, 'model.msg')
    status['message']['node_shape'] = node_shape
    status['message']['dimension'] = dimension
    status['message']['size'] = size
    status['message']['gap'] = gap
    status['message']['element_type'] = element_type
    status['message']['gap'] = gap
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(status['message'], f, ensure_ascii=False)
        
    filename = os.path.join(model_path, 'args.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(args[:-1], f, ensure_ascii=False)

    filename = os.path.join(model_path, 'model.log')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(status['log'])

    status['progress'] = 100
    status['status'] = 'Done'

    return 0


if __name__ == "__main__":
    gap = 0.0
    size = [[0, 0.1], [0, 0.1]]
    dimension = [s[1] for s in size]
    node_shape = [100, 100]
    element_type = 'CPE4T'
    model_path = ''
    output_path = ''
    status = {'status': 'Submit', 'log': '', 'progress': 0}
    args = (gap, size, dimension, node_shape, element_type, model_path, output_path, status)
    print(status)
    create_mesh(*args)
    print(status)
