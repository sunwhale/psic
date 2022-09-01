# -*- coding: utf-8 -*-
"""
2D：矩形区域填充圆形
3D：六面体区域填充球体
>4D: 超立方体区域填充超球
"""


from psic.packing_spheres_in_cube import create_model


if __name__ == "__main__":
    ncircle = 32
    size = [[0, 1], [0, 1]]
    gap = 0.0
    num_add = 1000
    max_iter = 100
    dt0 = 0.01
    dt_interval = 1000
    rayleigh_para = 20
    num_ball = 1200
    rad_min = 10
    rad_max = 100
    model_path = ''
    thread_id = 1
    status = {'status': 'Submit', 'log': '', 'progress': 0}
    args = (ncircle, size, gap, num_add, max_iter, dt0, dt_interval, rayleigh_para, num_ball, rad_min, rad_max, model_path, status)
    print(status)
    create_model(*args)
    print(status)
