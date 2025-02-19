import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
from scipy.spatial import KDTree
import CurvesGenerator.reeds_shepp as rsCurve
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle
import os
import pandas as pd
from casadi import *

def map_maze():
    # Build Map
    s = [10, 10, np.deg2rad(90)]
    g = [25, 20, np.deg2rad(90)]
    obstacleX, obstacleY = [], []
    ## Maze
    for i in range(51):
        obstacleX.append(i)
        obstacleY.append(0)

    for i in range(51):
        obstacleX.append(0)
        obstacleY.append(i)

    for i in range(51):
        obstacleX.append(i)
        obstacleY.append(50)

    for i in range(51):
        obstacleX.append(50)
        obstacleY.append(i)

    for i in range(10,20):
        obstacleX.append(i)
        obstacleY.append(30)

    for i in range(30,51):
        obstacleX.append(i)
        obstacleY.append(30)

    for i in range(0,31):
        obstacleX.append(20)
        obstacleY.append(i)

    for i in range(0,31):
        obstacleX.append(30)
        obstacleY.append(i)

    for i in range(40,50):
        obstacleX.append(15)
        obstacleY.append(i)

    for i in range(25,40):
        obstacleX.append(i)
        obstacleY.append(35)

    xmin = min(obstacleX)
    xmax = max(obstacleX)
    ymin = min(obstacleY)
    ymax = max(obstacleY)

    return xmin, xmax, ymin, ymax, obstacleX, obstacleY, s, g

def map_parking(type, config_map):
    obstacleX, obstacleY = [], []
    ## Parallel Parking Map
    y_min = config_map['map-params'][type]['y_min']
    y_max =  config_map['map-params'][type]['y_max']
    y_mid = config_map['map-params'][type]['y_mid']
    x_min = config_map['map-params'][type]['x_min']
    x_max = config_map['map-params'][type]['x_max']
    x_1 = config_map['map-params'][type]['x_1']
    x_2 = config_map['map-params'][type]['x_2']
    s = config_map['map-params'][type]['start'][:3]
    s[2] = np.deg2rad(s[2])
    g = config_map['map-params'][type]['goal'][:3]
    g[2] = np.deg2rad(g[2])

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(y_min, y_max):
        obstacleX.append(x_min)
        obstacleY.append(i)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in range(y_min, y_max):
        obstacleX.append(x_max)
        obstacleY.append(i)

    for i in range(x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in range(x_min,x_1):
        obstacleX.append(i)
        obstacleY.append(y_mid)

    for i in range(x_2,x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_mid)

    for i in range(y_min,y_mid):
        obstacleX.append(x_1 - 1)
        obstacleY.append(i)

    for i in range(y_min,y_mid):
        obstacleX.append(x_2)
        obstacleY.append(i)

    for i in range(x_1,x_2):
        obstacleX.append(i)
        obstacleY.append(y_min)

    return x_min, x_max, y_min, y_max, obstacleX, obstacleY, s, g

def map_fran(type, config_map, Car_obj, axes):
    obstacleX, obstacleY = [], []
    ## Parallel Parking Map
    y_min = config_map['map-params'][type]['y_min']
    y_max =  config_map['map-params'][type]['y_max']
    x_min = config_map['map-params'][type]['x_min']
    x_max = config_map['map-params'][type]['x_max']
    s = config_map['map-params'][type]['start'][:3]


    s[2] = np.deg2rad(s[2])
    parked_car = config_map['map-params'][type]['parked_car'][:3]
    parked_car[2] = np.deg2rad(parked_car[2])

    gx = parked_car[0] + 7
    gy = parked_car[1]
    gyaw = parked_car[2]

    g = (gx, gy, gyaw)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(y_min, y_max):
        obstacleX.append(x_min)
        obstacleY.append(i)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in range(y_min, y_max):
        obstacleX.append(x_max)
        obstacleY.append(i)

    for i in range(x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    car = np.array(
        [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(parked_car[2]), -math.sin(parked_car[2])],
                          [math.sin(parked_car[2]), math.cos(parked_car[2])]])
    car = np.dot(rotationZ, car)
    car1 = car + np.array([[parked_car[0]], [parked_car[1]]]) # (2xN) N are vertices

    ## Car

    for i in range(car1.shape[1]-1):
        obstacleX.append(car1[0, i])
        obstacleY.append(car1[1, i])

    for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
        obstacleX.append(car1[0, 0])
        obstacleY.append(i)

    for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
        obstacleY.append(car1[1, 1])
        obstacleX.append(i)

    for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
        obstacleX.append(car1[0, 2])
        obstacleY.append(i)

    for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
        obstacleY.append(car1[1, 3])
        obstacleX.append(i)

    car2 = car + np.array([[24.0], [parked_car[1]]])  # (2xN) N are vertices

    for i in range(car2.shape[1]-1):
        obstacleX.append(car2[0, i])
        obstacleY.append(car2[1, i])

    for i in range(math.floor(car2[1, 0]), math.ceil(car2[1, 1])-1, -1):
        obstacleX.append(car2[0, 0])
        obstacleY.append(i)

    for i in range(math.ceil(car2[0, 1]), math.floor(car2[0, 2])+1):
        obstacleY.append(car2[1, 1])
        obstacleX.append(i)

    for i in range(math.ceil(car2[1, 2]), math.floor(car2[1, 3])+1):
        obstacleX.append(car2[0, 2])
        obstacleY.append(i)

    for i in range(math.floor(car2[0, 3]), math.ceil(car2[0, 4]), -1):
        obstacleY.append(car2[1, 3])
        obstacleX.append(i)

    for ax in axes:
        ## Boundaries
        ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='black', linewidth=4)
        ax.plot(car1[0, :], car1[1, :], color='black', linewidth=2)
        ax.plot(car2[0, :], car2[1, :], color='black', linewidth=2)

    return x_min, x_max, y_min, y_max, obstacleX, obstacleY, s, g

def map_lot(type, config_map, Car_obj, axes):

    s = config_map['map-params'][type]['start'][:3]
    # s = [8, 20, 90]
    # s = (8.        , 23.        ,  1.57079633)
    # s = [6.87135284, 39.73844655, 2.35266696]
    s[2] = np.deg2rad(s[2])

    obstacleX, obstacleY = [], []
    ## Parallel Parking Map
    y_min = config_map['map-params'][type]['y_min']
    x_min = config_map['map-params'][type]['x_min']
    l_w = config_map['map-params'][type]['lane_width']
    p_w = config_map['map-params'][type]['park_width']
    p_l = config_map['map-params'][type]['park_length']
    n_r = config_map['map-params'][type]['n_rows']
    n_s = config_map['map-params'][type]['n_spaces']

    n_s1 = int(n_s/2)

    x_max = int(x_min + (n_r+1)*l_w + 2*n_r*p_l)
    y_max = int(y_min + l_w + n_s1 * p_w + l_w)

    ## goal
    g_x = x_min + 2 * l_w + 2 * p_l + p_l - 1.5
    g_y = y_min + l_w + 3 * p_w + int(p_w / 2)
    g_yaw = 180.0

    g_yaw = np.deg2rad(g_yaw)

    g = (g_x, g_y, g_yaw)

    # g = (16.48192011, 36.40054653, 0.20252271)
    # g = (27.27751797, 38.41901272, -0.18841261)

    center_line_park_row_y = []
    for i in range(y_min + l_w,int(y_min + l_w + n_s1*p_w)+1):
        center_line_park_row_y.append(i)

    center_line_park_row_x = x_min + l_w + p_l
    for _ in range(n_r):

        obstacleX = obstacleX + [center_line_park_row_x]*len(center_line_park_row_y)
        obstacleY = obstacleY + center_line_park_row_y

        for ax in axes:
            ax.plot([center_line_park_row_x]*len(center_line_park_row_y), center_line_park_row_y, color='black', linewidth=4)

        short_line_park_row_x = []
        for i in range(center_line_park_row_x - p_l, int(center_line_park_row_x + p_l) + 1):
            short_line_park_row_x.append(i)

        short_line_park_row_y = y_min + l_w + p_w
        for _ in range(n_s1-1):
            obstacleX = obstacleX + short_line_park_row_x
            obstacleY = obstacleY + [short_line_park_row_y]*len(short_line_park_row_x)
            for ax in axes:
                ax.plot(short_line_park_row_x, [short_line_park_row_y]*len(short_line_park_row_x), color='black',
                        linewidth=4)

            short_line_park_row_y += p_w

        center_line_park_row_x += int(2*p_l + l_w)

    # parked cars 0
    p_x = x_min + 1 * l_w + 1 * p_l - 1.5
    p_yaw = np.pi
    j_indices = [0, 1, 2, 3]

    car = np.array(
        [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                          [math.sin(p_yaw), math.cos(p_yaw)]])

    car = np.dot(rotationZ, car)

    for j in j_indices:

        p_y = y_min + l_w + j * p_w + p_w / 2

        car1 = car + np.array([[p_x], [p_y]]) # (2xN) N are vertices

        ## Car

        for i in range(car1.shape[1]-1):
            obstacleX.append(car1[0, i])
            obstacleY.append(car1[1, i])

        for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
            obstacleX.append(car1[0, 0])
            obstacleY.append(i)

        for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
            obstacleY.append(car1[1, 1])
            obstacleX.append(i)

        for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
            obstacleX.append(car1[0, 2])
            obstacleY.append(i)

        for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
            obstacleY.append(car1[1, 3])
            obstacleX.append(i)
        for ax in axes:
            ax.plot(car1[0, :], car1[1, :], color='black', linewidth=4)

    # parked cars 1
    p_x = x_min + 1 * l_w + 1 * p_l + 1.5
    p_yaw = 0.0
    j_indices = [1, 2, 3]

    car = np.array(
        [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                          [math.sin(p_yaw), math.cos(p_yaw)]])

    car = np.dot(rotationZ, car)

    for j in j_indices:

        p_y = y_min + l_w + j * p_w + p_w / 2

        car1 = car + np.array([[p_x], [p_y]]) # (2xN) N are vertices

        ## Car

        for i in range(car1.shape[1]-1):
            obstacleX.append(car1[0, i])
            obstacleY.append(car1[1, i])

        for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
            obstacleX.append(car1[0, 0])
            obstacleY.append(i)

        for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
            obstacleY.append(car1[1, 1])
            obstacleX.append(i)

        for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
            obstacleX.append(car1[0, 2])
            obstacleY.append(i)

        for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
            obstacleY.append(car1[1, 3])
            obstacleX.append(i)
        for ax in axes:
            ax.plot(car1[0, :], car1[1, :], color='black', linewidth=4)

    # parked cars 2
    p_x = x_min + 2 * l_w + 3 * p_l - 1.5
    p_yaw = np.pi
    j_indices = [1, 4]

    car = np.array(
        [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                          [math.sin(p_yaw), math.cos(p_yaw)]])

    car = np.dot(rotationZ, car)
    for j in  j_indices:

        p_y = y_min + l_w + j * p_w + p_w / 2

        car1 = car + np.array([[p_x], [p_y]]) # (2xN) N are vertices

        ## Car

        for i in range(car1.shape[1]-1):
            obstacleX.append(car1[0, i])
            obstacleY.append(car1[1, i])

        for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
            obstacleX.append(car1[0, 0])
            obstacleY.append(i)

        for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
            obstacleY.append(car1[1, 1])
            obstacleX.append(i)

        for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
            obstacleX.append(car1[0, 2])
            obstacleY.append(i)

        for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
            obstacleY.append(car1[1, 3])
            obstacleX.append(i)

        for ax in axes:
            ax.plot(car1[0, :], car1[1, :], color='black', linewidth=4)

    # parked cars 3
    p_x = x_min + 2 * l_w + 3 * p_l + 1.5
    p_yaw = 0.0
    j_indices = [0, 1, 3]

    car = np.array(
        [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                          [math.sin(p_yaw), math.cos(p_yaw)]])

    car = np.dot(rotationZ, car)
    for j in  j_indices:

        p_y = y_min + l_w + j * p_w + p_w / 2

        car1 = car + np.array([[p_x], [p_y]]) # (2xN) N are vertices

        ## Car

        for i in range(car1.shape[1]-1):
            obstacleX.append(car1[0, i])
            obstacleY.append(car1[1, i])

        for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
            obstacleX.append(car1[0, 0])
            obstacleY.append(i)

        for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
            obstacleY.append(car1[1, 1])
            obstacleX.append(i)

        for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
            obstacleX.append(car1[0, 2])
            obstacleY.append(i)

        for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
            obstacleY.append(car1[1, 3])
            obstacleX.append(i)

        for ax in axes:
            ax.plot(car1[0, :], car1[1, :], color='black', linewidth=4)

    # for ax in axes:
    #     ax.scatter(parked_cars_x + moving_cars_x, parked_cars_y + moving_cars_y, color='black',
    #             marker='s', s=6)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_min)
        obstacleY.append(i)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_max)
        obstacleY.append(i)

    for ax in axes:
        ## Boundaries
        ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='black', linewidth=4)


    return x_min, x_max, y_min, y_max, obstacleX, obstacleY, s, g

def map_angle(type, config_map, Car_obj, axes):

    obstacleX, obstacleY = [], []
    ## Parallel Parking Map
    y_min = config_map['map-params'][type]['y_min']
    x_min = config_map['map-params'][type]['x_min']
    l_w = config_map['map-params'][type]['lane_width']
    p_w = config_map['map-params'][type]['park_width']
    p_l = config_map['map-params'][type]['park_length']
    n_s = config_map['map-params'][type]['n_spaces']

    angle_deg = 55.0
    angle = np.deg2rad(angle_deg)

    start_x_park = x_min + 7
    goal_y_park_offset = 2

    x_max = math.ceil(start_x_park + (n_s+2)*p_w)
    y_max = math.ceil(y_min + 1*p_l*(np.sin(angle)) + 1*l_w)

    ## start
    s = (x_min + 2, y_min + p_l*(np.sin(angle)) + 3, 0.0)

    ## goal
    g_x = start_x_park + 1*p_w + 0.5*p_w + goal_y_park_offset/np.tan(angle)
    g_y = y_min + goal_y_park_offset
    g_yaw = angle

    g = (g_x, g_y, g_yaw)

    y_start = y_min
    y_end = y_min + p_l*(np.sin(angle))

    for i in range(n_s+1):

        x_start = start_x_park + i*p_w
        x_end = x_start + p_l*(np.cos(angle))

        points = np.linspace([x_start, y_start], [x_end, y_end], num=int(max(x_end - x_start + 1, y_end - y_start + 1)))

        obstacleX = obstacleX + points[:, 0].tolist()
        obstacleY = obstacleY + points[:, 1].tolist()

        for ax in axes:
            ax.plot(points[:, 0], points[:, 1], color='black', linewidth=4)

    # parked cars 0
    p_y = g_y
    p_yaw = angle
    j_indices = [0, 2, 3]

    car = np.array(
        [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                          [math.sin(p_yaw), math.cos(p_yaw)]])

    car = np.dot(rotationZ, car)

    for j in j_indices:

        p_x = start_x_park + 0.5*p_w + goal_y_park_offset/np.tan(angle) + j * p_w

        car1 = car + np.array([[p_x], [p_y]]) # (2xN) N are vertices

        ## Car

        for i in range(car1.shape[1]-1):
            obstacleX.append(car1[0, i])
            obstacleY.append(car1[1, i])

        for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
            obstacleX.append(car1[0, 0])
            obstacleY.append(i)

        for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
            obstacleY.append(car1[1, 1])
            obstacleX.append(i)

        for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
            obstacleX.append(car1[0, 2])
            obstacleY.append(i)

        for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
            obstacleY.append(car1[1, 3])
            obstacleX.append(i)
        for ax in axes:
            ax.plot(car1[0, :], car1[1, :], color='black', linewidth=4)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_min)
        obstacleY.append(i)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_max)
        obstacleY.append(i)

    for ax in axes:
        ## Boundaries
        ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='black', linewidth=4)


    return x_min, x_max, y_min, y_max, obstacleX, obstacleY, s, g

def map_angle_head_in(type, config_map, Car_obj, axes):

    obstacleX, obstacleY = [], []
    ## Parallel Parking Map
    y_min = config_map['map-params'][type]['y_min']
    x_min = config_map['map-params'][type]['x_min']
    l_w = config_map['map-params'][type]['lane_width']
    p_w = config_map['map-params'][type]['park_width']
    p_l = config_map['map-params'][type]['park_length']
    n_s = config_map['map-params'][type]['n_spaces']
    c_l = config_map['map-params'][type]['curb_length']
    r_d = config_map['map-params'][type]['row_depth']

    angle_deg = config_map['map-params'][type]['angle']
    angle = np.deg2rad(angle_deg)

    start_x_park = x_min + 10
    goal_y_park_offset = r_d

    x_max = math.ceil(start_x_park + r_d/np.tan(angle) + (n_s+2)*c_l)
    y_max = math.ceil(y_min + r_d + 1*l_w)

    ## start
    s = (x_min + 2, y_min + r_d + 4, 0.0)

    ## goal
    g_x = start_x_park + c_l/2 + 1*c_l
    g_y = y_min + 1
    g_yaw = -angle

    g = (g_x, g_y, g_yaw)

    y_start = y_min + r_d
    y_end = y_min

    for i in range(n_s+1):

        x_start = start_x_park + i*c_l
        x_end = x_start + r_d/np.tan(angle)

        points = np.linspace([x_start, y_start], [x_end, y_end], num=int(max(x_end - x_start + 1, y_end - y_start + 1)))

        obstacleX = obstacleX + points[:, 0].tolist()
        obstacleY = obstacleY + points[:, 1].tolist()

        for ax in axes:
            ax.plot(points[:, 0], points[:, 1], color='black', linewidth=2)

    # parked cars 0
    p_y = g_y
    p_yaw = -angle
    j_indices = [0, 2, 3, 4]

    car = np.array(
        [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                          [math.sin(p_yaw), math.cos(p_yaw)]])

    car = np.dot(rotationZ, car)

    for j in j_indices:

        p_x = start_x_park + 0.5*c_l + j * c_l

        car1 = car + np.array([[p_x], [p_y]]) # (2xN) N are vertices

        ## Car

        for i in range(car1.shape[1]-1):
            obstacleX.append(car1[0, i])
            obstacleY.append(car1[1, i])

        for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
            obstacleX.append(car1[0, 0])
            obstacleY.append(i)

        for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
            obstacleY.append(car1[1, 1])
            obstacleX.append(i)

        for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
            obstacleX.append(car1[0, 2])
            obstacleY.append(i)

        for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
            obstacleY.append(car1[1, 3])
            obstacleX.append(i)
        for ax in axes:
            ax.plot(car1[0, :], car1[1, :], color='black', linewidth=2)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_min)
        obstacleY.append(i)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_max)
        obstacleY.append(i)

    for ax in axes:
        ## Boundaries
        ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='black', linewidth=4)


    return x_min, x_max, y_min, y_max, obstacleX, obstacleY, s, g

class Car_class:
    def __init__(self, config_planner):
        self.maxSteerAngle = config_planner['vehicle-params']['max_steer_rate']
        self.steerPresion = config_planner['HA*-params']['steer_prec'] # number of steering inputs = 2*steerPresion + 1 within [-maxSteerAngle, maxSteerAngle]
        self.wheelBase = config_planner['vehicle-params']['wheelbase_length']
        self.axleToBack = config_planner['vehicle-params']['axle_to_back']
        self.axleToFront = self.wheelBase + self.axleToBack # assuming space between front axle and front of car = axle to back
        self.width = config_planner['vehicle-params']['vehicle_width']
        self.safety_margin = config_planner['HA*-params']['safety_margin']

class Cost_class:
    def __init__(self, config_planner):
        self.reverse = config_planner['HA*-params']['cost']['reverse']
        self.directionChange = config_planner['HA*-params']['cost']['directionChange']
        self.steerAngle = config_planner['HA*-params']['cost']['steerAngle']
        self.steerAngleChange = config_planner['HA*-params']['cost']['steerAngleChange']
        self.hybridCost = config_planner['HA*-params']['cost']['hybridCost'] # heuristic cost
        self.stationary = config_planner['HA*-params']['cost']['stationary']

class RSCost_class:
    def __init__(self, config_planner):
        self.reverse = config_planner['HA*-params']['RScost']['reverse']
        self.directionChange = config_planner['HA*-params']['RScost']['directionChange']
        self.steerAngle = config_planner['HA*-params']['RScost']['steerAngle']
        self.steerAngleChange = config_planner['HA*-params']['RScost']['steerAngleChange']

class Node:
    def __init__(self, gridIndex, traj, obst_traj, steeringAngle, direction, cost, parentIndex, rs_flag = 0):
        self.gridIndex = gridIndex         # grid block x, y, yaw index, time
        self.traj = traj                   # trajectory x, y, yaw from parentNode to currentNode
        self.steeringAngle = steeringAngle # steering angle throughout the trajectory
        self.direction = direction         # direction throughout the trajectory
        self.cost = cost                   # node cost
        self.parentIndex = parentIndex     # parent node index
        self.obst_traj = obst_traj  # obstacle trajectory from parentNode's time to currentNode's time
        self.rs_flag = rs_flag

    def index(self):
        # Index is a tuple consisting (x, y, yaw, t)
        return tuple([self.gridIndex[0], self.gridIndex[1], self.gridIndex[2], self.gridIndex[3]])

class HolonomicNode:
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex
        self.cost = cost
        self.parentIndex = parentIndex

    def index(self):
        # Index is a tuple consisting grid index, used for checking if two nodes are near/same
        return tuple([self.gridIndex[0], self.gridIndex[1]])

class MapParameters:
    def __init__(self, xmin, xmax, ymin, ymax, obstacleX, obstacleY, xyResolution, yawResolution):

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        # calculate min max map grid index
        self.mapMinX = math.floor(xmin / xyResolution)
        self.mapMaxX = math.floor(xmax/ xyResolution)
        self.mapMinY = math.floor(ymin / xyResolution)
        self.mapMaxY = math.floor(ymax / xyResolution)
        self.mapMaxYaw = math.floor(2*np.pi / yawResolution)
        self.xyResolution = xyResolution     # grid block length
        self.yawResolution = yawResolution  # grid block possible yaws
        self.ObstacleKDTree = KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)]) # KDTree representating obstacles
        self.obstacleX = obstacleX           # Obstacle x coordinate list
        self.obstacleY = obstacleY           # Obstacle y coordinate list

        self.map_coords_yaw = [[[(xyResolution/2 + i*xyResolution, xyResolution/2 + j*xyResolution, k*yawResolution) for k in range(self.mapMaxYaw)] for j in range(self.mapMaxY)] for i in range(self.mapMaxX)]

        self.map_coords = [[(xyResolution / 2 + i * xyResolution, xyResolution / 2 + j * xyResolution) for j in range(self.mapMaxY)] for i in
                           range(self.mapMaxX)]

    # Obstacle occupancy in the gridworld according to xyResolution
    def obstaclesMap(self):

        # Compute Grid Index for obstacles
        obstacleX = [math.floor(x / self.xyResolution) for x in self.obstacleX]
        obstacleY = [math.floor(y / self.xyResolution) for y in self.obstacleY]

        # Set all Grid locations to No Obstacle
        obstacles = [[False for i in range(self.mapMinY, self.mapMaxY + 1)] for i in range(self.mapMinX, self.mapMaxX + 1)]

        # Set Grid Locations with obstacles to True
        for x in range(self.mapMinX, self.mapMaxX + 1):
            for y in range(self.mapMinY, self.mapMaxY + 1):
                for i, j in zip(obstacleX, obstacleY):
                    if math.hypot(i - x, j - y) <= 1 / 2:
                        obstacles[x][y] = True  # Shouldn't it be [x][y]???
                        break

        return obstacles

class HA_planner:
    def __init__(self, type, config_map, config_planner, axes):
        self.Car = Car_class(config_planner)
        self.Cost = Cost_class(config_planner)
        self.RSCost = RSCost_class(config_planner)

        self.cost_thresh_RS = config_planner['HA*-params']['cost_thresh_RS']

        if type=='maze':
            xmin, xmax, ymin, ymax, obstacleX, obstacleY, s, g = map_maze()
        elif type=='reverse_lot':
            xmin, xmax, ymin, ymax, obstacleX, obstacleY, s, g = map_lot(type, config_map, self.Car, axes)
        elif type=='parallel_fran':
            xmin, xmax, ymin, ymax, obstacleX, obstacleY, s, g = map_fran(type, config_map, self.Car, axes)
        elif type=='angle':
            xmin, xmax, ymin, ymax, obstacleX, obstacleY, s, g = map_angle_head_in(type, config_map, self.Car, axes)
        else:
            xmin, xmax, ymin, ymax, obstacleX, obstacleY, s, g = map_parking(type, config_map)

        # default start and goal locations. Can change in run.
        self.s = s
        self.g = g

        self.Map = MapParameters(xmin, xmax, ymin, ymax, obstacleX, obstacleY, config_planner['HA*-params']['xy_res'], np.deg2rad(config_planner['HA*-params']['yaw_res']))
        self.obstacles_occupancy = self.Map.obstaclesMap()

        # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
        self.holonomicMotionCommands = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

        # Motion commands for a Non-Holonomic Robot like a Car or Bicycle (Trajectories using Steer Angle and Direction)
        direction = 1
        self.motionCommands = []
        self.motionCommands.append([0, 0])

        for i in np.arange(self.Car.maxSteerAngle, -(self.Car.maxSteerAngle + self.Car.maxSteerAngle/self.Car.steerPresion), -self.Car.maxSteerAngle/self.Car.steerPresion):
            self.motionCommands.append([i, direction])
            self.motionCommands.append([i, -direction])

        self.sim_length_expand = config_planner['HA*-params']['sim_length_expand']
        self.dt = config_planner['HA*-params']['dt']

        # Dynamic obstacles
        dynamic_x_0 = config_planner['dynamic_obst']['x_0']
        dynamic_y_0 = config_planner['dynamic_obst']['y_0']
        dynamic_vel_x = config_planner['dynamic_obst']['vel_x']
        dynamic_vel_y = config_planner['dynamic_obst']['vel_y']
        self.dynamic_xy_0 = np.array([dynamic_x_0, dynamic_y_0]).T
        self.dynamic_xy = np.array([dynamic_x_0, dynamic_y_0]).T
        self.dynamic_vel = np.array([dynamic_vel_x, dynamic_vel_y]).T

        ## End counter
        self.counter_end = config_planner['HA*-params']['counter_end']

    def kinematicSimulationNode(self, currentNode, motionCommand, traj_obst):

        # Simulate node using given current Node and Motion Commands
        traj = []
        sim_length = int((self.sim_length_expand/self.dt))-1
        if motionCommand[1] == 0:
            # angle = rsCurve.pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * self.dt / self.Car.wheelBase * math.tan(motionCommand[0]))
            single = [currentNode.traj[-1][0], currentNode.traj[-1][1], currentNode.traj[-1][2]]
            traj = [single] * (sim_length+1)
        else:
            # angle = rsCurve.pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * self.dt / self.Car.wheelBase * math.tan(motionCommand[0]))
            traj.append([currentNode.traj[-1][0] + motionCommand[1] * self.dt * math.cos(currentNode.traj[-1][2]),
                        currentNode.traj[-1][1] + motionCommand[1] * self.dt * math.sin(currentNode.traj[-1][2]),
                        rsCurve.pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * self.dt / self.Car.wheelBase * math.tan(motionCommand[0]))])
            for i in range(sim_length):
                traj.append([traj[i][0] + motionCommand[1] * self.dt * math.cos(traj[i][2]),
                            traj[i][1] + motionCommand[1] * self.dt * math.sin(traj[i][2]),
                            rsCurve.pi_2_pi(traj[i][2] + motionCommand[1] * self.dt / self.Car.wheelBase * math.tan(motionCommand[0]))])

        # Find grid index
        gridIndex = [math.floor(traj[-1][0]/self.Map.xyResolution),
                     math.floor(traj[-1][1]/self.Map.xyResolution),
                     math.floor(traj[-1][2]/self.Map.yawResolution)]

        # Check if node is valid
        if not self.isValid(traj, traj_obst, gridIndex):
            return None

        # Calculate Cost of the node
        cost = self.simulatedPathCost(currentNode, motionCommand, self.sim_length_expand)

        nodeIndex = gridIndex + [math.floor((currentNode.index()[-1] + self.dt * sim_length)/self.dt)]

        return Node(nodeIndex, traj, traj_obst, motionCommand[0], motionCommand[1], cost, currentNode.index())

    def reedsSheppNode(self, currentNode, goalNode, end_flag):

        # Get x, y, yaw of currentNode and goalNode
        startX, startY, startYaw = currentNode.traj[-1][0], currentNode.traj[-1][1], currentNode.traj[-1][2]
        goalX, goalY, goalYaw = goalNode.traj[-1][0], goalNode.traj[-1][1], goalNode.traj[-1][2]

        # Instantaneous Radius of Curvature
        radius = math.tan(self.Car.maxSteerAngle)/self.Car.wheelBase

        #  Find all possible reeds-shepp paths between current and goal node
        reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, self.dt)

        # # Check if reedsSheppPaths is empty
        # if not reedsSheppPaths:
        #     return None, None
        #
        # # Find path with lowest cost considering non-holonomic constraints
        # costQueue = heapdict()
        # for path in reedsSheppPaths:
        #     cost = self.reedsSheppCost(currentNode, path)
        #     if end_flag == 0 and cost < 800:
        #         costQueue[path] = cost
        #     elif end_flag == 1:
        #         costQueue[path] = cost
        #
        # dt = self.dt
        # # Find first path in priority queue that is collision free
        # while len(costQueue)!=0:
        #     path = costQueue.popitem()[0]
        #     traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(1,len(path.x))]
        #     traj_obst = [currentNode.obst_traj[-1] + i * dt * self.dynamic_vel for i in range(1, len(path.x))]
        #     traj_mod = []
        #     if (not self.collision(traj)) and (not self.dynamic_collision(traj, traj_obst)):
        #         for i in range(1, len(path.directions) - 1):
        #             if path.directions[i] * path.directions[i + 1] < 0:
        #                 traj_mod = traj_mod + [[path.x[i], path.y[i], path.yaw[i]]]
        #             traj_mod.append([path.x[i], path.y[i], path.yaw[i]])
        #         traj_mod.append([path.x[-1], path.y[-1], path.yaw[-1]])
        #         goalNode.gridIndex[-1] = currentNode.index()[-1] + dt*(len(path.x)-1)
        #         goalNode.parentIndex =  currentNode.index()
        #         goalNode.traj = traj_mod
        #         goalNode.obst_traj = traj_obst
        #         goalNode.cost = cost
        #         return Node(goalNode.gridIndex, traj, traj_obst, None, None, cost, currentNode.index()), traj_obst
        #
        # return None, None

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths:
            return None, None, None, None, None


        # Find path with lowest cost considering non-holonomic constraints
        costQueue = heapdict()
        for path in reedsSheppPaths:
            traj = [[path.x[k], path.y[k], path.yaw[k]] for k in range(1, len(path.x))]
            traj_obst = [currentNode.obst_traj[-1] + i * self.dt * self.dynamic_vel for i in range(1, len(path.x))]
            if (not self.collision(traj)) and (not self.dynamic_collision(traj, traj_obst)):
                cost = self.reedsSheppCost(currentNode, path)
                if end_flag == 0 and cost < 800:
                    costQueue[path] = cost
                elif end_flag == 1:
                    costQueue[path] = cost

        # Find first path in priority queue that is collision free
        while len(costQueue)!=0:
            path = costQueue.popitem()[0]
            traj_mod = []
            traj_obst = [currentNode.obst_traj[-1] + i * self.dt * self.dynamic_vel for i in range(1, len(path.x))]
            for i in range(1, len(path.directions)):
                traj_mod.append([path.x[i], path.y[i], path.yaw[i]])
            cost = self.reedsSheppCost(currentNode, path)
            goalNode.gridIndex[-1] = math.floor((currentNode.index()[-1] + self.dt*(len(path.x)-1))/self.dt)
            goalNode.parentIndex =  currentNode.index()
            goalNode.traj = traj_mod
            goalNode.obst_traj = traj_obst
            goalNode.cost = cost
            goalNode.rs_flag = 1
            return goalNode, traj_obst, path.L, path.lengths, path.ctypes

        return None, None, None, None, None

    def isValid(self, traj, traj_obst, gridIndex):

        # Check if Node is out of map bounds
        if gridIndex[0]<=self.Map.mapMinX or gridIndex[0]>=self.Map.mapMaxX or \
           gridIndex[1]<=self.Map.mapMinY or gridIndex[1]>=self.Map.mapMaxY:
            return False

        # Check if Node is colliding with an obstacle
        if self.collision(traj):
            return False

        if self.dynamic_collision(traj, traj_obst):
            return False

        return True

    def collision(self, traj):

        carRadius = (self.Car.axleToFront + self.Car.axleToBack)/2  # along longitudinal axis
        # carRadius_query = np.sqrt((carRadius + Car.safety_margin)**2 + (Car.width / 2 + Car.safety_margin)**2)
        carRadius_query = carRadius + 1.5*self.Car.safety_margin
        dl = (self.Car.axleToFront - self.Car.axleToBack)/2
        # to get center of car
        for i in traj:
            cx = i[0] + dl * math.cos(i[2])
            cy = i[1] + dl * math.sin(i[2])
            pointsInObstacle = self.Map.ObstacleKDTree.query_ball_point([cx, cy], carRadius_query)

            if not pointsInObstacle:
                continue

            for p in pointsInObstacle:
                xo = self.Map.obstacleX[p] - cx
                yo = self.Map.obstacleY[p] - cy
                dx = xo * math.cos(i[2]) + yo * math.sin(i[2]) # from center to obstacle along longitudinal axis
                dy = -xo * math.sin(i[2]) + yo * math.cos(i[2]) # from center to obstacle along lateral axis

                if abs(dx) < (carRadius + self.Car.safety_margin) and abs(dy) < (self.Car.width / 2 + self.Car.safety_margin):
                    return True

        return False

    def dynamic_collision(self, traj, traj_obst):

        carRadius = (self.Car.axleToFront + self.Car.axleToBack)/2  # along longitudinal axis
        carRadius_query = carRadius + 2.5*self.Car.safety_margin
        dl = (self.Car.axleToFront - self.Car.axleToBack)/2
        # to get center of car
        for i, o in zip(traj, traj_obst):
            cx = i[0] + dl * math.cos(i[2])
            cy = i[1] + dl * math.sin(i[2])
            pointsInObstacle = np.where(np.linalg.norm(np.array([cx, cy]) - o, axis=1) < carRadius_query)[0]

            if not len(pointsInObstacle):
                continue

            for p in pointsInObstacle:
                xo = o[p, 0] - cx
                yo = o[p, 1] - cy
                dx = xo * math.cos(i[2]) + yo * math.sin(i[2]) # from center to obstacle along longitudinal axis
                dy = -xo * math.sin(i[2]) + yo * math.cos(i[2]) # from center to obstacle along lateral axis

                if abs(dx) < (carRadius + 2*self.Car.safety_margin) and abs(dy) < (self.Car.width / 2 + 2*self.Car.safety_margin):
                    return True

        return False

    def reedsSheppCost(self, currentNode, path):

        # Previos Node Cost
        cost = currentNode.cost

        # Distance cost
        for i in path.lengths:
            if i >= 0:
                cost += i
            else:
                cost += abs(i) * self.RSCost.reverse

        # Direction change cost
        for i in range(len(path.lengths)-1):
            if path.lengths[i] * path.lengths[i+1] < 0:
                cost += self.RSCost.directionChange

        # Steering Angle Cost
        for i in path.ctypes:
            # Check types which are not straight line
            if i!="S":
                cost += self.Car.maxSteerAngle * self.RSCost.steerAngle

        # Steering Angle change cost
        turnAngle=[0.0 for _ in range(len(path.ctypes))]
        for i in range(len(path.ctypes)):
            if path.ctypes[i] == "R":
                turnAngle[i] = - self.Car.maxSteerAngle
            if path.ctypes[i] == "WB":
                turnAngle[i] = self.Car.maxSteerAngle

        for i in range(len(path.lengths)-1):
            cost += abs(turnAngle[i+1] - turnAngle[i]) * self.RSCost.steerAngleChange

        return cost

    def simulatedPathCost(self, currentNode, motionCommand, simulationLength):

        # Previous Node Cost
        cost = currentNode.cost

        # Time cost
        if motionCommand[1] > 0:
            cost += simulationLength
        elif motionCommand[1] == 0:
            cost += simulationLength * self.Cost.stationary
        else:
            cost += simulationLength * self.Cost.reverse

        # Direction change cost
        if currentNode.direction * motionCommand[1] == -1:
            cost += self.Cost.directionChange

        # Steering Angle Cost
        cost += abs(motionCommand[0]) * self.Cost.steerAngle  # Shouldn't it be abs(motionCommand[0])?

        # Steering Angle change cost
        cost += abs(motionCommand[0] - currentNode.steeringAngle) * self.Cost.steerAngleChange

        return cost

    def cost_holonomic_command(self, holonomicMotionCommand):
        # Compute Eucledian Distance between two nodes
        return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

    def holonomicNodeIsValid(self, neighbourNode):

        # Check if Node is out of map bounds
        if neighbourNode.gridIndex[0]<= self.Map.mapMinX or \
           neighbourNode.gridIndex[0]>= self.Map.mapMaxX or \
           neighbourNode.gridIndex[1]<= self.Map.mapMinY or \
           neighbourNode.gridIndex[1]>= self.Map.mapMaxY:
            return False

        # Check if Node on obstacle
        if self.obstacles_occupancy[neighbourNode.gridIndex[0]][neighbourNode.gridIndex[1]]:
            return False

        return True

    def holonomicCostsWithObstacles(self, goalNode):

        gridIndex = [math.floor(goalNode.traj[-1][0]/self.Map.xyResolution), math.floor(goalNode.traj[-1][1]/self.Map.xyResolution)]
        gNode = HolonomicNode(gridIndex, 0, tuple(gridIndex))

        openSet = {gNode.index(): gNode}
        closedSet = {}

        priorityQueue =[]
        heapq.heappush(priorityQueue, (gNode.cost, gNode.index()))

        while True:
            if not openSet:
                break

            _, currentNodeIndex = heapq.heappop(priorityQueue)
            currentNode = openSet[currentNodeIndex]
            openSet.pop(currentNodeIndex)
            closedSet[currentNodeIndex] = currentNode

            for i in range(len(self.holonomicMotionCommands)):
                neighbourNode = HolonomicNode([currentNode.gridIndex[0] + self.holonomicMotionCommands[i][0],
                                          currentNode.gridIndex[1] + self.holonomicMotionCommands[i][1]],
                                          currentNode.cost + self.cost_holonomic_command(self.holonomicMotionCommands[i]), currentNodeIndex)

                if not self.holonomicNodeIsValid(neighbourNode):
                    continue

                neighbourNodeIndex = neighbourNode.index()

                if neighbourNodeIndex not in closedSet:
                    if neighbourNodeIndex in openSet:
                        if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                            openSet[neighbourNodeIndex].cost = neighbourNode.cost
                            openSet[neighbourNodeIndex].parentIndex = neighbourNode.parentIndex
                            # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                    else:
                        openSet[neighbourNodeIndex] = neighbourNode
                        heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))

        holonomicCost = [[np.inf for i in range(self.Map.mapMinY, self.Map.mapMaxY)]for i in range(self.Map.mapMinX, self.Map.mapMaxX)]

        for nodes in closedSet.values():
            holonomicCost[nodes.gridIndex[0]][nodes.gridIndex[1]]=nodes.cost

        return holonomicCost

    def backtrack(self, startNode, goalNode, closedSet):

        # print("Current Node Index: ", goalNode.gridIndex)
        a, b, c = zip(*goalNode.traj)
        x=a[::-1]
        y=b[::-1]
        yaw=c[::-1]
        t = [goalNode.index()[-1]]
        obst_xn = [point[:, 0] for point in goalNode.obst_traj]
        obst_yn = [point[:, 1] for point in goalNode.obst_traj]
        obst_x = obst_xn[::-1]
        obst_y = obst_yn[::-1]
        currentNodeIndex = goalNode.parentIndex
        currentNode = closedSet[currentNodeIndex]

        startNodeIndex = startNode.index()

        # Iterate till we reach start node from goal node
        while currentNodeIndex != startNodeIndex:
            a, b, c = zip(*currentNode.traj)
            obst_xn = [point[:, 0] for point in currentNode.obst_traj]
            obst_yn = [point[:, 1] for point in currentNode.obst_traj]
            obst_x += obst_xn[::-1]
            obst_y += obst_yn[::-1]
            x += a[::-1]
            y += b[::-1]
            yaw += c[::-1]
            t += [currentNodeIndex[-1]]
            currentNodeIndex = currentNode.parentIndex
            currentNode = closedSet[currentNodeIndex]
            # print("Current Node Index: ", currentNodeIndex)
        return x[::-1], y[::-1], yaw[::-1], t[::-1], obst_x[::-1], obst_y[::-1]

    def heuristic(self, node1, node2):
        # Get x, y, yaw of currentNode and goalNode
        startX, startY, startYaw = node1.traj[-1][0], node1.traj[-1][1], node1.traj[-1][2]
        goalX, goalY, goalYaw = node2.traj[-1][0], node2.traj[-1][1], node2.traj[-1][2]

        # Instantaneous Radius of Curvature
        radius = math.tan(self.Car.maxSteerAngle) / self.Car.wheelBase

        #  Find all possible reeds-shepp paths between current and goal node
        reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 1)

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths:
            return None

        # Find path with lowest cost considering non-holonomic constraints
        costQueue = heapdict()
        for path in reedsSheppPaths:
            costQueue[path] = self.reedsSheppCost(node1, path)
        path = costQueue.popitem()[0]
        cost = self.reedsSheppCost(node1, path)
        return cost

        # return np.linalg.norm(np.array(node1.gridIndex[:2]) - np.array(node2.gridIndex[:2]))

    def run(self, s, g, holonomicHeuristics, rs_flag, end_flag, axes, count, end_sim):
        # dynamic_xy_traj = [self.dynamic_xy + self.dt*self.dynamic_vel]
        # Compute Grid Index for start and Goal node

        rs_path = 0
        sGridIndex = [math.floor(s[0] / self.Map.xyResolution),
                      math.floor(s[1] / self.Map.xyResolution),
                      math.floor(s[2]/self.Map.yawResolution),
                      0]
        gGridIndex = [math.floor(g[0] / self.Map.xyResolution),
                      math.floor(g[1] / self.Map.xyResolution),
                      math.floor(g[2]/self.Map.yawResolution),
                      None]

        # Create start and end Node
        startNode = Node(sGridIndex, [s], [self.dynamic_xy], 0, 0, 0 , tuple(sGridIndex))
        goalNode = Node(gGridIndex, [g], [], 0, 0, 0, tuple(gGridIndex))

        # Find Holonomic Heuristric
        # holonomicHeuristics = holonomicCostsWithObstacles(goalNode, mapParameters)

        # Add start node to open Set
        openSet = {startNode.index(): startNode}
        closedSet = {}

        # Create a priority queue for acquiring nodes based on their cost's
        costQueue = heapdict()

        # Add start node into priority queue
        # costQueue[startNode.index()] = max(startNode.cost , self.Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]])
        costQueue[startNode.index()] = startNode.cost + self.Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]]

        # costQueue[startNode.index()] = startNode.cost + self.Cost.hybridCost * self.heuristic(startNode, goalNode)
        counter = 0
        xr, yr, yawr = [], [], []
        # Run loop while path is found or open set is empty
        currentTime = 0
        while True:
            counter +=1
            # Check if openSet is empty, if empty no solution available
            if not openSet or counter>=self.counter_end:
                # print("No path!")
                return [], [], [], [], counter, [], [], xr, yr, yawr, rs_path # x, y, yaw, counter, obst_x, obst_y, xr, yr, yawr

            # Get first node in the priority queue
            currentNodeIndex = costQueue.popitem()[0]
            currentNode = openSet[currentNodeIndex]

            # Revove currentNode from openSet and add it to closedSet
            openSet.pop(currentNodeIndex)
            closedSet[currentNodeIndex] = currentNode

            # Get Reed-Shepp Node if available
            if rs_flag:
                # hol_cost = holonomicHeuristics[currentNode.gridIndex[0]][currentNode.gridIndex[1]]
                # hol_cost = np.linalg.norm(np.array(currentNode.traj[-1]) - np.array(goalNode.traj[-1]))
                rSNode, rsObst_path, path_length, lengths, ctype = self.reedsSheppNode(currentNode, goalNode, end_flag)
                # rSNode, rsObst_path = self.reedsSheppNode(currentNode, goalNode, end_flag)

                # If Reeds-Shepp Path is found exit
                if rSNode:
                    # print("Path found with RS!")
                    closedSet[rSNode.index()] = rSNode
                    xr, yr, yawr = zip(*rSNode.traj)
                    rs_path = 1
                    break

            # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
            if currentNodeIndex[:-1] == goalNode.index()[:-1]:
                # print("Path found without RS!")
                xr, yr, yawr = [], [], []
                goalNode.gridIndex[-1] = currentNode.index()[-1]
                goalNode.parentIndex =  currentNode.parentIndex
                goalNode.traj = currentNode.traj
                goalNode.obst_traj = currentNode.obst_traj
                goalNode.cost = currentNode.cost
                # print(currentNode.traj[-1])
                break

            # Get all simulated Nodes from current node
            sim_length = int((self.sim_length_expand / self.dt)-1)
            first_neighbor = 0
            traj_obst = [currentNode.obst_traj[-1] + i * self.dt * self.dynamic_vel for i in range(1,sim_length+2)]
            ## Draw obstacle's trajectory
            # obst_xn = [point[:, 0] for point in traj_obst]
            # obst_yn = [point[:, 1] for point in traj_obst]
            # if count==end_sim:
            #     for ax in axes:
            #         ax.plot(obst_xn, obst_yn, linewidth=0.3, color='r', alpha=0.3)

            for i in range(len(self.motionCommands)):
                simulatedNode = self.kinematicSimulationNode(currentNode, self.motionCommands[i], traj_obst)

                # Check if path is within map bounds and is collision free
                if not simulatedNode:
                    continue
                elif simulatedNode and first_neighbor == 0: # for efficient costQueue maintenance
                    currentTime = currentTime + self.dt * sim_length
                    first_neighbor = 1

                ## To draw Simulated Node (x, y, z)
                # x,y,z =zip(*simulatedNode.traj)
                # if count == end_sim:
                #     for ax in axes:
                #         ax.plot(x, y, linewidth=0.3, color='g', alpha=0.3)

                # Check if simulated node is already in closed set
                simulatedNodeIndex = simulatedNode.index()

                # if simulatedNodeIndex[:-1] == goalNode.index()[:-1]:
                #     print("Path found without RS! Simulated")
                #     xr, yr, yawr = [], [], []
                #     closedSet[simulatedNodeIndex] = simulatedNode
                #     goalNode.gridIndex[-1] = simulatedNode.index()[-1]
                #     goalNode.parentIndex = simulatedNode.parentIndex
                #     goalNode.traj = simulatedNode.traj
                #     goalNode.obst_traj = simulatedNode.obst_traj
                #     goalNode.cost = simulatedNode.cost
                #     # Backtrack
                #     x, y, yaw, obst_x, obst_y = self.backtrack(startNode, goalNode, closedSet)
                #
                #     return x, y, yaw, counter, obst_x, obst_y, xr, yr, yawr

                temp_cost = simulatedNode.cost + self.Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]]
                # temp_cost = max(simulatedNode.cost, self.Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])

                if simulatedNodeIndex not in closedSet:
                    # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                    if simulatedNodeIndex not in openSet:
                        openSet[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = temp_cost
                    else:
                        if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
                            openSet[simulatedNodeIndex] = simulatedNode
                            costQueue[simulatedNodeIndex] = temp_cost

            ## Plotting obstacle trajectory while exploring nodes
            # print("pause!")
            # dynamic_xy_next = self.dynamic_xy + (int((self.sim_length_expand / self.dt)) - 1) * self.dt * self.dynamic_vel
            #
            # # if count==end_sim:
            # #     ax.plot(self.dynamic_xy[:, 0], self.dynamic_xy[:, 1], linestyle='', marker='o', markersize=12, color='r')
            # #     ax.plot(dynamic_xy_next[:, 0], dynamic_xy_next[:, 1], linestyle='', marker='o', markersize=12,
            # #             color='r')
            #
            # self.dynamic_xy = self.dynamic_xy + (int((self.sim_length_expand / self.dt))) * self.dt * self.dynamic_vel
            #
            # for _ in range(int((self.sim_length_expand / self.dt))):
            #     dynamic_xy_traj.append(dynamic_xy_traj[-1] + self.dt * self.dynamic_vel)

        # Backtrack
        x, y, yaw, t, obst_x, obst_y = self.backtrack(startNode, goalNode, closedSet)

        return x, y, yaw, t, counter, obst_x, obst_y, xr, yr, yawr, rs_path

    def eval_metrics(self, x, y, yaw):
        traj_rear = np.array([x, y, yaw]).T
        traj_vel_sign = np.diff(traj_rear, axis=0) / self.dt
        traj_vel = np.abs(np.diff(traj_rear, axis=0)) / self.dt
        traj_acc = np.abs(np.diff(traj_vel, axis=0)) / self.dt
        traj_vel[:, 2] = np.arctan2(np.sin(traj_vel_sign[:, 2]), np.cos(traj_vel_sign[:, 2]))*180/np.pi
        traj_vel_mean = np.mean(traj_vel, axis=0)
        traj_vel_max = np.max(traj_vel, axis=0)
        traj_vel_std = np.std(traj_vel, axis=0)
        traj_vel_metrics = [traj_vel_mean, traj_vel_std, traj_vel_max]

        traj_acc_mean = np.mean(traj_acc, axis=0)
        traj_acc_max = np.max(traj_acc, axis=0)
        traj_acc_std = np.std(traj_acc, axis=0)
        traj_acc_metrics = [traj_acc_mean, traj_acc_std, traj_acc_max]

        nodes_explored = len(x)
        min_d_obst_all =[]

        carRadius = (self.Car.axleToFront + self.Car.axleToBack)/2  # along longitudinal axis
        dl = (self.Car.axleToFront - self.Car.axleToBack)/2 # to get center of car
        traj_center_xy = traj_rear[:, :2] + dl * np.hstack((np.cos(traj_rear[:, 2]).reshape((-1, 1)), np.sin(traj_rear[:, 2]).reshape((-1, 1))))
        obst_ind_all = self.Map.ObstacleKDTree.query_ball_point(traj_center_xy, r=carRadius + 2 * self.Car.safety_margin)
        obst_ind_filter = [(index, element) for index, element in enumerate(obst_ind_all) if element]
        min_d_obst = np.inf
        for traj_ind, obst_ind  in obst_ind_filter:
            xo = np.array([self.Map.obstacleX[i] for i in obst_ind]) - traj_center_xy[traj_ind, 0]
            yo = np.array([self.Map.obstacleY[i] for i in obst_ind]) - traj_center_xy[traj_ind, 1]
            dx = xo * math.cos(traj_rear[traj_ind, 2]) + yo * math.sin(traj_rear[traj_ind, 2]) # from center to obstacle along longitudinal axis
            dy = -xo * math.sin(traj_rear[traj_ind, 2]) + yo * math.cos(traj_rear[traj_ind, 2]) # from center to obstacle along lateral axis
            closeness_dx = np.maximum(abs(dx) - carRadius, 0)
            closeness_dy = np.maximum(abs(dy) - self.Car.width/2, 0)
            min_d_obst_all.append([traj_ind, np.min(np.sqrt(closeness_dx**2 + closeness_dy**2))])
            min_d_obst = min(min_d_obst_all[-1][-1], min_d_obst)

        path_length = np.sum(np.linalg.norm(np.diff(traj_rear[:, :2], axis=0), axis=1))

        return path_length, nodes_explored, min_d_obst_all, min_d_obst, traj_vel_metrics, traj_acc_metrics

class MPC_control:
    def __init__(self, config_planner):
        self.N = config_planner['MPC-params']['N']
        self.w_pos = config_planner['MPC-params']['w_pos']
        self.w_angles = config_planner['MPC-params']['w_angles']
        self.ind_epsilon = config_planner['MPC-params']['closest_ind_eps']
        self.wheelBase = config_planner['vehicle-params']['wheelbase_length']
        self.max_speed = config_planner['vehicle-params']['max_speed']
        self.max_steer = config_planner['vehicle-params']['max_steer_rate']
        self.dt = config_planner['MPC-params']['dt']
        self.dt_HA = config_planner['HA*-params']['dt']

    def get_spline(self, p1, p2):
        """
        Get spline from start point p1 to goal point p2

        input:
            p1: start point [x,y,heading]
            p1: goal point [x,y,heading]
        return:
            list: in the form of [[x,y,0],...]
        """

        x1, y1, theta1 = p1[0], p1[1], p1[2]
        x2, y2, theta2 = p2[0], p2[1], p2[2]

        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > 1e-3:

            dx1 = np.cos(theta1)
            dy1 = np.sin(theta1)
            dx2 = np.cos(theta2)
            dy2 = np.sin(theta2)

        else:
            dx1, dy1, dx2, dy2 = 0, 0, 0, 0

        t = np.linspace(0, self.dt_HA, int(self.dt_HA/self.dt))
        t0 = t[0]
        t1 = t[-1]

        # Matrix form to be inversed
        A = np.asarray(
            [
                [1, t0, t0**2, t0**3],  # x  @ 0
                [0, 1, 2 * t0, 3 * t0**2],  # x' @ 0
                [1, t1, t1**2, t1**3],  # x  @ 1
                [0, 1, 2 * t1, 3 * t1**2],  # x' @ 1
            ]
        )

        # Compute for X
        X = np.asarray([x1, dx1, x2, dx2]).transpose()
        bx = np.linalg.solve(A, X)

        # Compute for Y
        Y = np.asarray([y1, dy1, y2, dy2]).transpose()
        by = np.linalg.solve(A, Y)

        x = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(), bx)
        y = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(), by)
        psi = np.append(theta1, np.arctan2(y[1:]-y[:-1], x[1:]-x[:-1]))
        traj = [[xx, yy, ps] for xx, yy, ps in zip(x, y, psi)]

        return traj

    def xdot_func(self, x, u):
        ## u = (vel, steering rate)
        xdot = np.array(
            [u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), (u[0] / self.wheelBase) * np.tan(u[1])])
        return xdot

    def sim_dynamics(self, x, u):
        dt = self.dt
        # Runge-Kutta 4 integration
        k1 = self.xdot_func(x, u)
        k2 = self.xdot_func(x+dt / 2 * k1, u)
        k3 = self.xdot_func(x + dt / 2 * k2, u)
        k4 = self.xdot_func(x + dt * k3, u)
        xnext = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # ## second-order Runge-Kutta method
        # xdot_t = xdot_func(x, u)
        # xdot_t1 = xdot_func(x + 0.5*dt*xdot_t, u)
        # theta = x[2] + xdot_t1[2]*dt
        # theta_next = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize to -pi to pi
        # x_next = x[0] + xdot_t1[0]*dt
        # y_next = x[1] + xdot_t1[1] * dt
        # xnext = np.array([x_next, y_next, theta_next])

        # xnext = x + dt_m * k1

        theta = np.copy(xnext[2])
        while theta > np.pi:
            theta -= 2.0 * np.pi

        while theta < -np.pi:
            theta += 2.0 * np.pi

        xnext[2] = theta

        return xnext


    def run(self, x_t, x_star):
        closest_ind = np.argmin(np.linalg.norm(x_t[:2] - x_star[:, :2], axis=1))
        goal_ind_start = np.minimum(closest_ind+1, x_star.shape[0] - 1)
        goals_MPC = x_star[goal_ind_start:]
        N_act = goals_MPC.shape[0]
        dim_x = x_t.shape[0]

        opti = casadi.Opti()

        X = opti.variable(dim_x,N_act+1)
        U = opti.variable(2,N_act)
        def f(x, u):
            return casadi.vertcat(u[0]*casadi.cos(x[2]), u[0]*casadi.sin(x[2]), (u[0]/self.wheelBase)*casadi.tan(u[1]))

        J = 0
        dt_m = self.dt
        for k in range(N_act):
            # Runge-Kutta 4 integration
            k1 = f(X[:, k], U[:, k])
            k2 = f(X[:, k] + dt_m / 2 * k1, U[:, k])
            k3 = f(X[:, k] + dt_m / 2 * k2, U[:, k])
            k4 = f(X[:, k] + dt_m * k3, U[:, k])
            x_next = X[:, k] + dt_m / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)
            J += (self.w_pos*dot(X[:2, k + 1]-goals_MPC[k, :2],X[:2, k + 1]-goals_MPC[k, :2])) # +self.w_angles*dot(X[2:, k + 1]-goals_MPC[k, 2:],X[2:, k + 1]-goals_MPC[k, 2:]))

        opti.minimize(J)
        opti.subject_to(X[:, 0] == x_t)
        opti.subject_to(U[0, :] <= self.max_speed)
        opti.subject_to(U[0, :] >= -self.max_speed)  # speed limit
        opti.subject_to( U[1, :] <= 0.6)
        opti.subject_to( U[1, :] >= -0.6)  # speed limit

        opti.set_initial(U, np.zeros((2, N_act)))
        opts = {"ipopt.print_level":0, "ipopt.sb": "yes", "print_time": False}
        opti.solver('ipopt', opts)
        sol = opti.solve()
        if N_act <= 1:
            u_t = sol.value(U).reshape((-1,1))
        else:
            u_t = sol.value(U)

        return u_t

def drawCar(Car_obj, x, y, yaw):
    car = np.array([[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
                    [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
    car = np.dot(rotationZ, car)
    car += np.array([[x], [y]])
    return car

def init_plot(axes, figs, HA, staticX, staticY, dynamic_x, dynamic_y, time_traj, goal_MPC, goal_all_xy, HA_path_MPC):

    # Get the colormap
    cmap = plt.get_cmap('rainbow')
    # Define how many colors you want
    num_colors = len(time_traj)
    # Generate a list of RGBA values from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    car = drawCar(HA.Car, HA.s[0], HA.s[1], HA.s[2])
    car_g = drawCar(HA.Car, HA.g[0], HA.g[1], HA.g[2])

    for ax in axes:
        ax.plot(car[0, :], car[1, :], color='blue', linewidth=2, label='Start')
        ax.plot(car_g[0, :], car_g[1, :], color='green', linewidth=2, label='Goal')
        # Draw Start, Goal Location Map and Path
        ax.arrow(HA.s[0], HA.s[1], 1 * math.cos(HA.s[2]), 1 * math.sin(HA.s[2]), width=.1)
        ax.arrow(HA.g[0], HA.g[1], 1 * math.cos(HA.g[2]), 1 * math.sin(HA.g[2]), width=.1)
        ax.set_xlim(HA.Map.xmin, HA.Map.xmax)
        ax.set_ylim(HA.Map.ymin, HA.Map.ymax)
        ax.plot(goal_all_xy[:, 0], goal_all_xy[:, 1], linestyle='', marker='o', color=colors[0], zorder=0, label='Global Path')
        # ax.set_title("Hybrid A*")
        ax.set_aspect('equal')

    ## Only for animation
    car_plot_a, = axes[1].plot(car[0, :], car[1, :], color='brown', linewidth=1, label='Current state')
    arrow_plot_a = axes[1].annotate("", xy=(HA.s[0] + 1 * math.cos(HA.s[2]), HA.s[1] + 1 * math.sin(HA.s[2])),
                                xytext=(HA.s[0], HA.s[1]), arrowprops=dict(arrowstyle="simple"))
    props = dict(boxstyle='round', facecolor='w', alpha=0.5, edgecolor='black', linewidth=2)
    time_text = 't=' + str(time_traj[0])
    text_t  = axes[1].text(1,33, time_text, fontsize=26, bbox=props)
    HA_plot_a, = axes[1].plot(HA_path_MPC[0][:, 0], HA_path_MPC[0][:, 1], linestyle='-', marker='o', color='magenta', label='Local Path')
    HA_goal_plot_a, = axes[1].plot(goal_MPC[0][0], goal_MPC[0][1], linestyle='', marker='o', color='blue')

    ## Initial dynamic obstacle
    circles = []
    for i in range(HA.dynamic_xy_0.shape[0]):
        if i==0:
            circle = Circle((HA.dynamic_xy_0[i, 0], HA.dynamic_xy_0[i, 1]), HA.Car.safety_margin, facecolor='red',  label='Dynamic obstacle')
        else:
            circle = Circle((HA.dynamic_xy_0[i, 0], HA.dynamic_xy_0[i, 1]), HA.Car.safety_margin, facecolor='red')
        circles.append(circle)
        axes[1].add_patch(circle)

    figs[0].suptitle('Path', fontsize=20)
    figs[1].suptitle('Animation', fontsize=20)
    figs[2].suptitle('Car', fontsize=20)

    return axes, figs, car_plot_a, arrow_plot_a, circles, text_t, HA_plot_a, HA_goal_plot_a

def plot_anim(axes, figs, car_plot_a, arrow_plot_a, dynamic_plot_a, HA_plot_a, HA_goal_plot_a, text_t, Car_obj, state_all_traj, time_traj, obst_x, obst_y, HA_path_MPC, goal_MPC, save_file_name):
    # Define the update function for the animation

    axc = axes[2]
    ## Animation with car displayed at each time step
    for k in range(len(state_all_traj)):
        # plt.cla()
        car = drawCar(Car_obj,state_all_traj[k][0], state_all_traj[k][1], state_all_traj[k][2])
        if k == 0:
            label_text = 'Current state'
        else:
            label_text = ''
        axc.plot(car[0, :], car[1, :], color='brown', linewidth = 1, alpha=0.2, label=label_text)
        # drawCar(x[k], y[k], yaw[k], ax)
        # axc.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
        # axa.pause(0.01)

    extend_end = 5
    state_all_traj = state_all_traj + [state_all_traj[-1]]*extend_end
    HA_path_MPC = HA_path_MPC + [HA_path_MPC[-1]] * extend_end
    goal_MPC = goal_MPC + [goal_MPC[-1]] * extend_end

    obst_x = obst_x + [obst_x[-1]]*extend_end
    obst_y = obst_y + [obst_y[-1]] * extend_end
    time_traj = np.hstack((time_traj, [time_traj[-1]]*extend_end))

    def update(frame):

        car = drawCar(Car_obj, state_all_traj[frame][0], state_all_traj[frame][1], state_all_traj[frame][2])
        car_plot_a.set_data(car[0, :], car[1, :])
        arrow_plot_a.xy = [state_all_traj[frame][0] + 1 * math.cos(state_all_traj[frame][2]),
                           state_all_traj[frame][1] + 1 * math.sin(state_all_traj[frame][2])]
        arrow_plot_a.set_position((state_all_traj[frame][0], state_all_traj[frame][1]))
        HA_plot_a.set_data(HA_path_MPC[frame][:, 0], HA_path_MPC[frame][:, 1])
        HA_goal_plot_a.set_data(goal_MPC[frame][0], goal_MPC[frame][1])
        text_t.set_text('t=' + str(time_traj[frame]))
        for i in range(len(dynamic_plot_a)):
            # Update circle positions
            dynamic_plot_a[i].center = (obst_x[frame][i], obst_y[frame][i],)
        # dynamic_plot_a.set_data(obst_x[frame], obst_y[frame])
        plot_list = dynamic_plot_a + [car_plot_a] + [arrow_plot_a] + [text_t] + [HA_plot_a] + [HA_goal_plot_a]
        return plot_list

    axes[1].legend(loc='upper right', fontsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)

    axes[0].legend(loc='upper right', fontsize=6)
    axes[0].tick_params(axis='both', which='major', labelsize=6)

    axes[2].legend(loc='upper right', fontsize=6)
    axes[2].tick_params(axis='both', which='major', labelsize=6)

    # Create the animation
    figa = figs[1]
    ani = FuncAnimation(figa, update, frames=len(state_all_traj), blit=True, interval=200, repeat_delay = 1000)

    file_dir_anim =  save_file_name + '.mp4'
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=-1)
    ani.save(file_dir_anim, writer=writer)

    figs[0].savefig(save_file_name + '.pdf')
    figs[2].savefig(save_file_name + '_car' + '.pdf') # 1 is animation

    plt.show()

def save_results(x, y, yaw, time_traj, save_file_name):

    HA_path_array = np.array([x, y, yaw, time_traj]).T
    path = save_file_name + '.npy'
    with open(path, 'wb') as f:
        np.save(f, HA_path_array)

    ## Save simulation results in an Excel file for varying steering precision

    # # Column names
    # columns = ["Steering precision", "Average computation time", "STD computation time", "Path length", "Number of nodes", "Number of HA* explorations", "Closest distance to obstacle"]
    #
    # # Create a dictionary from lists and column names
    # data = {
    #     columns[0]: steering_precs,
    #     columns[1]: time_steer,
    #     columns[2]: time_steer_std,
    #     columns[3]: path_length_steer,
    #     columns[4]: nodes_steer,
    #     columns[5]: counter_steer,
    #     columns[6]: close_to_obst_steer
    # }
    # # Create a DataFrame
    # df = pd.DataFrame(data)
    #
    # # Save to Excel
    # path = save_file_name + '.xlsx'
    # df.to_excel(path, index=False)

def main(type, config_map, config_planner, end_sim, start_ind_time):

    ## Figure for animation and plot
    figa, axa = plt.subplots()
    fig, ax = plt.subplots()
    figc, axc = plt.subplots()
    axes = [ax, axa, axc]
    figs = [fig, figa, figc]

    # steering_precs = np.hstack((2, np.arange(5, 25, 5))) # to run Hybrid A star for varying steering precision

    home_path = os.path.abspath(os.getcwd())

    ## To load the init path
    sim_results_path = home_path + '/Sim_results/'
    path = sim_results_path + 'HA_star_path/Init/' + 'g1_all_cars_init_reverse_lot_5cost_thresh_RS.npy'

    with open(path, 'rb') as f:
        goal_all = np.load(f)

    load_rs = sim_results_path + '/RS_path_adaptive_paper.npy'
    with open(load_rs, 'rb') as f:
        goal_rs_all = np.load(f)

    goal_all_xy = goal_all[:, :2]
    goal_all_f = goal_all[:, -1]

    ## Create the planner object
    HA = HA_planner(type, config_map, config_planner, axes)
    MPC = MPC_control(config_planner)
    N_look_HA = config_planner['HA*-params']['N_look_HA']

    staticX = HA.Map.obstacleX.copy()
    staticY = HA.Map.obstacleY.copy()

    # Add new obstacles
    dynamic_x = [] #config_planner['obstacle']['center_x']
    dynamic_y = [] #config_planner['obstacle']['center_y']
    obstacleX = HA.Map.obstacleX + dynamic_x
    obstacleY = HA.Map.obstacleY + dynamic_y

    ## Goal point
    goal_threshold_pos = config_planner['goal_thresh_pos']
    goal_threshold_head = config_planner['goal_thresh_head']
    goal = HA.g
    goal_xy = goal[:2]

    ## Save simulation details and metrics
    time_all = []
    time_MPC_all = []
    path_length_all=[]
    nodes_all = []
    closeness_to_obst_all = []
    counter_all_sim=[]
    traj_vel_metrics_all = []
    traj_acc_metrics_all = []
    goal_ind_all_sim = []
    HA_path_all_sim = []

    # axes, figs, car_plot_a, arrow_plot_a, circles, text_t, HA_plot_a, HA_goal_plot_a = init_plot(axes, figs, HA, staticX, staticY, dynamic_x, dynamic_y, [0.0], [np.array(HA.s)], goal_all_xy, [np.array(HA.s).reshape((1,-1))])
    # HA_path, = ax.plot(HA.s[0], HA.s[1], linestyle='', marker='o', color='magenta')
    # start_plot, = ax.plot(HA.s[0], HA.s[1], linestyle='', marker='o', color='purple')
    # goal_plot, = ax.plot(HA.s[0], HA.s[1], linestyle='', marker='o', color='pink')
    # ax.plot(HA.dynamic_xy[:, 0], HA.dynamic_xy[:, 1], linestyle='', marker='o', color='red')
    ## Start simulation
    for count in range(end_sim+start_ind_time):

        ## Initialize vehicle
        state_t = np.array(HA.s) + np.array(config_planner['disturb']) # with some disturbance
        # state_t = goal_all[-2, :3]
        xy_t = state_t[:2]
        reached_goal = (np.linalg.norm(xy_t - goal_xy) < goal_threshold_pos)

        # map for new obstacles
        HA.Map = MapParameters(HA.Map.xmin, HA.Map.xmax, HA.Map.ymin, HA.Map.ymax, obstacleX, obstacleY,
                               HA.Map.xyResolution, HA.Map.yawResolution)

        ## Dynamic obstacles' initial point
        dynamic_xy = HA.dynamic_xy_0
        HA.dynamic_xy = dynamic_xy

        ## save simulation results
        goal_ind_all = []
        HA_path_all = []
        counter_all = []
        HA_r_path_all = []
        RS_flag_all = []
        obst_x_all = [HA.dynamic_xy[:, 0]]
        obst_y_all = [HA.dynamic_xy[:, 1]]
        # obst_x_all = []
        # obst_y_all = []
        actual_t = 0.0
        actual_t_all = [actual_t]
        state_all = [state_t]
        u_all = []
        iter_all =[]
        HA_path_MPC = [state_t.reshape((1,-1))]
        goal_MPC = [state_t]
        # g = np.array([0, 0, 0])
        count_end = 0
        end_flag = 0
        while not reached_goal:

            ## Get current goal using some look-ahead (go at-least look_dist)

            closest_ind = min(np.argmin(np.linalg.norm(xy_t - goal_all_xy, axis=1)), goal_all_xy.shape[0]-1)
            # goal_ind = np.copy(closest_ind)
            # dist_cum = np.linalg.norm(xy_t - goal_all_xy[goal_ind])
            # while dist_cum <= look_dist and goal_ind < goal_all_xy.shape[0]-1:
            #     dist_cum += np.linalg.norm(goal_all_xy[goal_ind] - goal_all_xy[min(goal_ind+1, goal_all_xy.shape[0]-1)])
            #     goal_ind += 1

            # # get center of vehicle at goal
            # carRadius = (HA.Car.axleToFront + HA.Car.axleToBack) / 2  # along longitudinal axis
            # dl = (HA.Car.axleToFront - HA.Car.axleToBack) / 2  # to get center of car
            # g_c = np.array([g[0] + dl * math.cos(g[2]), g[1] + dl * math.sin(g[2])])
            #
            # # decide the current goal that is obstacle free
            # dist_to_obst, _ = HA.Map.ObstacleKDTree.query(g_c)
            #
            # while dist_to_obst <= carRadius + HA.Car.safety_margin:
            #     goal_ind = np.minimum(goal_ind + 1, goal_all.shape[0] - 1)
            #     g = goal_all[goal_ind]
            #     g_c = np.array([g[0] + dl * math.cos(g[2]), g[1] + dl * math.sin(g[2])])
            #     dist_to_obst, _ = HA.Map.ObstacleKDTree.query(g_c)

            # goal_ind = -1 # just using the given goal if no re-planning (like choosing an intermediate goal)
            goal_ind = min(closest_ind + N_look_HA, goal_all_xy.shape[0] - 1)
            s = state_t

            # Run Hybrid A*
            x = []
            iter = -1
            goal_ind = goal_ind + 1
            max_iter_HA = N_look_HA
            # print("closest index", closest_ind)
            # print("Closest Goal", goal_all[closest_ind, :3])
            start_t = time.time()
            while len(x) == 0 and iter < max_iter_HA:
                iter += 1
                if goal_ind == goal_all_xy.shape[0] - 1 and count_end > 20:
                    goal_ind = goal_all_xy.shape[0] - 1
                    HA.counter_end = 1000
                    end_flag = 1
                else:
                    goal_ind = max(closest_ind, goal_ind - 1)
                    # end_flag = 0
                # print("goal index", goal_ind)
                # print("State", state_t)
                # print("Goal", goal_all[goal_ind, :3])
                if goal_ind == closest_ind:
                    x = [state_t[0]]
                    y = [state_t[1]]
                    yaw = [state_t[2]]
                    xr = []
                    yr = []
                    yawr = []
                    rs_p = 0
                    counter = 1
                    break
                else:
                    g = goal_all[goal_ind, :3]
                    gGridIndex = [math.floor(g[0] / HA.Map.xyResolution),
                                  math.floor(g[1] / HA.Map.xyResolution),
                                  math.floor(g[2] / HA.Map.yawResolution)]

                    goalNode = Node(gGridIndex, [g], [HA.dynamic_xy], 0, 1, 0, tuple(gGridIndex))

                    # Find Holonomic Heuristric
                    holonomicHeuristics = HA.holonomicCostsWithObstacles(goalNode)
                    rs_flag = 1
                    x, y, yaw, t, counter, obst_x, obst_y, xr, yr, yawr, rs_p = HA.run(s, g, holonomicHeuristics, rs_flag, end_flag, [axa], count, end_sim)

            if HA.counter_end < 500:
                time_all.append(time.time() - start_t)

            goal_ind_all.append(goal_ind)
            iter_all.append(iter)

            if len(x) == 0:
                print("No path found!")
                print("Start", s)
                print("Goal", g)
                time_all.append(time.time() - start_t)
                counter_all.append(counter)
                break

            counter_all.append(counter)
            RS_flag_all.append(rs_p)
            HA_path_all.append(np.array([x, y, yaw]).T)
            HA_r_path_all.append(np.array([xr, yr, yawr]).T)
            # obst_x_all = obst_x_all + obst_x
            # obst_y_all = obst_y_all + obst_y
            start_MPC =time.time()
            x_star_goals = np.vstack((state_t, np.array([x, y, yaw]).T))
            term_ind = min(3, x_star_goals.shape[0] - 1)
            if goal_ind == goal_all.shape[0]-1:
                term_ind = x_star_goals.shape[0]-1
            for j in range(1, term_ind+1):
                time_interval = HA.dt
                state_t = x_star_goals[j, :]
                # state_MPC.append(state_t)
                state_all.append(state_t)
                HA_path_MPC.append(HA_path_all[-1])
                goal_MPC.append(g)
                HA.dynamic_xy = HA.dynamic_xy + HA.dynamic_vel * HA.dt
                # obst_x_MPC = obst_x_MPC + [HA.dynamic_xy[:, 0]]
                # obst_y_MPC = obst_y_MPC + [HA.dynamic_xy[:, 1]]
                obst_x_all = obst_x_all + [HA.dynamic_xy[:, 0]]
                obst_y_all = obst_y_all + [HA.dynamic_xy[:, 1]]
                actual_t += time_interval
                actual_t_all.append(actual_t)

            # state_all.append(state_MPC)
            # obst_x_all.append(obst_x_MPC)
            # obst_y_all.append(obst_y_MPC)
            # ax.plot(state_t[0], state_t[1], linestyle='', marker='o', color='blue')
            # ax.plot(obst_x, obst_y, linestyle='', marker='o', color='red')
            # np.array([obst_x[-1], obst_y[-1]]).T
            # ax.plot(HA.dynamic_xy[:, 0], HA.dynamic_xy[:, 1], linestyle='', marker='o', color='red')
            time_MPC_all.append(time.time()-start_MPC)
            # if abs(goal_all.shape[0] - goal_ind) == 2:
            #     count_end += 1
            #     if count_end > 10:
            #         reached_goal = 1
            #     xy_t = state_t[:2]
            if abs(goal_all.shape[0] - goal_ind) == 2:
                end_flag = 1
                count_end += 1
                xy_t = state_t[:2]
                reached_goal = (np.linalg.norm(xy_t - goal_xy) < goal_threshold_pos) and (np.linalg.norm(rsCurve.pi_2_pi(state_t[2] - goal[2])) < goal_threshold_head)
                if count_end > 100:
                    reached_goal = 1
            else:
                # end_flag = 0
                # state_t = np.array([x[-1], y[-1], yaw[-1]]) # assuming perfect control, the vehicle goes to the end of the path
                xy_t = state_t[:2]
                reached_goal = (np.linalg.norm(xy_t - goal_xy) < goal_threshold_pos) and (np.linalg.norm(rsCurve.pi_2_pi(state_t[2] - goal[2])) < goal_threshold_head)

        HA_path_array = np.vstack(HA_path_all)
        state_array = np.vstack(state_all)
        x=HA_path_array[:, 0]
        y=HA_path_array[:, 1]
        yaw = HA_path_array[:, 2]
        path_length, nodes, closeness_to_obst, min_closeness_to_obst, traj_vel_metrics, traj_acc_metrics = HA.eval_metrics(state_array[:, 0], state_array[:, 1],state_array[:, 2])
        path_length_all.append(path_length)
        closeness_to_obst_all.append(min_closeness_to_obst)
        nodes_all.append(nodes)
        counter_all_sim.append(counter_all)
        goal_ind_all_sim.append(goal_ind_all)
        traj_vel_metrics_all.append(traj_vel_metrics)
        traj_acc_metrics_all.append(traj_acc_metrics)

    avg_time = sum(time_all[start_ind_time:])/len(time_all[start_ind_time:])
    std_time = np.std(np.array(time_all[start_ind_time:]))

    avg_MPC_time = sum(time_MPC_all[start_ind_time:])/len(time_MPC_all[start_ind_time:])
    std_MPC_time = np.std(np.array(time_MPC_all[start_ind_time:]))

    print("Average computation time:", avg_time)
    print("Standard deviation of computation time:", std_time)
    print("Max computation time:", max(time_all[start_ind_time:]))
    print("Min computation time:", min(time_all[start_ind_time:]))
    print("Average MPC computation time:", avg_MPC_time)
    print("Standard deviation of MPC computation time:", std_MPC_time)
    print("Max MPC computation time:", max(time_MPC_all[start_ind_time:]))
    print("Min MPC computation time:", min(time_MPC_all[start_ind_time:]))
    print("Path length:", sum(path_length_all[start_ind_time:])/len(path_length_all[start_ind_time:]))
    print("Number of nodes:", nodes_all[-1])
    print("Counter of Run function:", counter_all_sim[-1])
    print("RS path:", RS_flag_all)
    print("Minimum closest distance to obstacle:", closeness_to_obst_all[-1])
    print("Max velocity [x, y, yaw]: ", traj_vel_metrics[2])
    print("Mean velocity [x, y, yaw]: ", traj_vel_metrics[0])
    print("STD velocity [x, y, yaw]: ", traj_vel_metrics[1])
    print("Max acc [x, y, yaw]: ", traj_acc_metrics[2])
    print("Mean acc [x, y, yaw]: ", traj_acc_metrics[0])
    print("STD acc [x, y, yaw]: ", traj_acc_metrics[1])

    actual_t_all = np.arange(len(state_all))
    ## Plot and save the results
    home_path = os.path.abspath(os.getcwd())
    sim_results_path = home_path + '/Sim_results/'
    save_file_name = sim_results_path + '/HA_star_adaptive_goal_dynamic_' + str(dynamic_xy.shape[0]) + '_' + type

    axes, figs, car_plot_a, arrow_plot_a, circles, text_t, HA_plot_a, HA_goal_plot_a = init_plot(axes, figs, HA, staticX, staticY, dynamic_x, dynamic_y, actual_t_all, goal_MPC, goal_rs_all, HA_path_MPC)
    save_results(state_array[:, 0], state_array[:, 1], state_array[:, 2], actual_t_all, save_file_name)
    plot_anim(axes, figs, car_plot_a, arrow_plot_a, circles, HA_plot_a, HA_goal_plot_a, text_t, HA.Car, state_all, actual_t_all, obst_x_all, obst_y_all, HA_path_MPC, goal_MPC, save_file_name)