"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import heapq
import math
import os
import sys
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from scipy.spatial import cKDTree
import time
from shapely.geometry import Point, Polygon

from timebudget import timebudget
from multiprocessing import Pool
import multiprocessing
# import ray

# ray.init(ignore_reinit_error=True)  # Initialize Ray

from itertools import chain

try:
    from parksim.path_planner.hybrid_astar.dynamic_programming_heuristic import calc_distance_heuristic
    import parksim.path_planner.hybrid_astar.reeds_shepp_path_planning as rs
    from parksim.path_planner.hybrid_astar.car import move, check_car_collision, MAX_STEER, WB, plot_car, plot_other_car, plot_car_trans
except Exception:
    raise

XY_GRID_RESOLUTION = 2  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(10.0)  # [rad]
MOTION_RESOLUTION = 0.5  # [m] path interpolate resolution
N_STEER = 3  # number of steer command
VR = 0.8  # robot radius

## FOV
long_front = 15.0
long_back = 5.0
lateral_bound = 12.0

num_points = 10
long_points = np.linspace(-long_back, long_front, num=num_points).reshape((-1,1))
lateral_points = np.linspace(-lateral_bound, lateral_bound, num=num_points).reshape((-1,1))

rect_points = np.vstack((np.hstack((long_points, np.array([-lateral_bound]*num_points).reshape((-1,1)))),
                         np.hstack((np.array([long_front]*num_points).reshape((-1,1)), lateral_points)),
                         np.hstack((long_points, np.array([lateral_bound]*num_points).reshape((-1,1)))),
                         np.hstack((np.array([-long_back]*num_points).reshape((-1,1)), lateral_points))
                         ))


# SB_COST = 100.0  # switch back penalty cost
SB_COST = 1.0  # switch back penalty cost
# BACK_COST = 5.0  # backward penalty cost
BACK_COST = 1.0  # backward penalty cost
# STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_CHANGE_COST = 0.5  # steer angle change penalty cost
STEER_COST = 0.5  # steer angle change penalty cost
H_COST = 1.0  # Heuristic cost

show_animation = False


class Node:

    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


class Path:

    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list #
        self.cost = cost


class Config:

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)

def calc_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        for d in [2, -2]:
            yield [steer, d]


def get_neighbors(current, config, ox, oy, kd_tree):
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list = [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)

    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l

    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, [d],
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    if n1.x_index == n2.x_index \
            and n1.y_index == n2.y_index \
            and n1.yaw_index == n2.yaw_index:
        return True
    return False

def analytic_expansion(current, goal, ox, oy, kd_tree):
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analytic_expansion(current, goal,
                                        c, ox, oy, kd_tree):
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c)

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        return True, f_path

    return False, None


def calc_rs_path_cost(reed_shepp_path):
    cost = 0.0
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in range(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost

# @ray.remote
def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    config = Config(tox, toy, xy_resolution, yaw_resolution)

    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, VR)

    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),
                        calc_index(start_node, config)))
    final_path = None

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return [], [], []

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        is_updated, final_path = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree)

        if is_updated:
            # print("path found")
            break

        for neighbor in get_neighbors(current, config, ox, oy,
                                      obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, final_path)
    return path


def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False


def calc_index(node, c):
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind

def hybrid_a_star_plotting(start, goal, path: Path, ox, oy, show_animation=False):
    plt.figure()

    plt.plot(ox, oy, ".k")
    rs.plot_arrow(start[0], start[1], start[2], fc='g')
    rs.plot_arrow(goal[0], goal[1], goal[2])

    plt.grid(True)
    plt.axis("equal")

    if isinstance(path, Path):
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list

        plt.plot(x, y, "-r", label="Hybrid A* path")

        if show_animation:
            for i_x, i_y, i_yaw in zip(x, y, yaw):
                plt.cla()
                plt.plot(ox, oy, ".k")
                plt.plot(x, y, "-r", label="Hybrid A* path")
                plt.grid(True)
                plt.axis("equal")
                plot_car(i_x, i_y, i_yaw)
                plt.pause(0.01)

class Car_class:
    def __init__(self, config_planner):
        self.maxSteerAngle = config_planner['vehicle-params']['max_steer']
        self.maxSpeed = config_planner['vehicle-params']['max_speed']
        self.velPresion = config_planner['HA*-params']['vel_prec']
        self.steerPresion = config_planner['HA*-params']['steer_prec'] # number of steering inputs = 2*steerPresion + 1 within [-maxSteerAngle, maxSteerAngle]
        self.wheelBase = config_planner['vehicle-params']['wheelbase_length']
        self.axleToBack = config_planner['vehicle-params']['axle_to_back']
        self.axleToFront = self.wheelBase + self.axleToBack # assuming space between front axle and front of car = axle to back
        self.length = self.axleToFront + self.axleToBack
        self.width = config_planner['vehicle-params']['vehicle_width']
        self.safety_margin = config_planner['HA*-params']['safety_margin']

def map_lot(type, config_map, Car_obj):

    s = config_map['map-params'][type]['start'][:3]
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

    s[0] = s[0] + l_w + 2*p_l + 0.75*l_w
    s[1] = s[1] + 0.2*l_w

    n_s1 = int(n_s/2)

    x_max = int(x_min + (n_r+1)*l_w + 2*n_r*p_l)
    y_max = int(y_min + l_w + n_s1 * p_w + l_w)

    center_line_park_row_y = [y_min + l_w, y_max - l_w]

    center_line_park_row_x_1 = x_min + l_w + p_l

    for _ in range(n_r):
        center_line_park_row_x = [center_line_park_row_x_1] * len(center_line_park_row_y)

        short_line_park_row_x = [center_line_park_row_x_1- p_l, center_line_park_row_x_1 + p_l]

        short_line_park_row_y_1 = y_min + l_w + p_w
        for _ in range(n_s1-1):
            short_line_park_row_y = [short_line_park_row_y_1]*len(short_line_park_row_x)

            short_line_park_row_y_1 += p_w

        center_line_park_row_x_1 += 2*p_l + l_w

    # parked cars 0
    p_x = x_min + 1 * l_w + p_l / 2 + Car_obj.length / 2 - Car_obj.axleToBack
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
            line = np.linspace(car1[:, i], car1[:, i+1])
            obstacleX = obstacleX + line[:, 0].tolist()
            obstacleY = obstacleY + line[:, 1].tolist()

        ## Car
        # for i in range(car1.shape[1]-1):
        #     obstacleX.append(car1[0, i])
        #     obstacleY.append(car1[1, i])
        #
        # for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
        #     obstacleX.append(car1[0, 0])
        #     obstacleY.append(i)
        #
        # for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
        #     obstacleY.append(car1[1, 1])
        #     obstacleX.append(i)
        #
        # for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
        #     obstacleX.append(car1[0, 2])
        #     obstacleY.append(i)
        #
        # for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
        #     obstacleY.append(car1[1, 3])
        #     obstacleX.append(i)

    # parked cars 1
    p_x = x_min + 1 * l_w + 1 * p_l + p_l / 2 - (Car_obj.length / 2 - Car_obj.axleToBack)
    p_yaw = 0.0
    j_indices = [0, 2, 3]

    for j in j_indices:

        if j==2:
            p_yaw = 10.0*np.pi/180
            p_y = y_min + l_w + j * p_w + p_w / 2 - 0.1 # -0.1 is disturbance
        else:
            p_yaw = 0.0
            p_y = y_min + l_w + j * p_w + p_w / 2

        ## Car
        car = np.array(
            [[-Car_obj.axleToBack, -Car_obj.axleToBack, Car_obj.axleToFront, Car_obj.axleToFront, -Car_obj.axleToBack],
             [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

        rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                              [math.sin(p_yaw), math.cos(p_yaw)]])

        car = np.dot(rotationZ, car)

        car1 = car + np.array([[p_x], [p_y]]) # (2xN) N are vertices

        # ## Car
        for i in range(car1.shape[1]-1):
            line = np.linspace(car1[:, i], car1[:, i+1])
            obstacleX = obstacleX + line[:, 0].tolist()
            obstacleY = obstacleY + line[:, 1].tolist()

        # for i in range(car1.shape[1]-1):
        #     obstacleX.append(car1[0, i])
        #     obstacleY.append(car1[1, i])
        #
        # for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
        #     obstacleX.append(car1[0, 0])
        #     obstacleY.append(i)
        #
        # for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
        #     obstacleY.append(car1[1, 1])
        #     obstacleX.append(i)
        #
        # for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
        #     obstacleX.append(car1[0, 2])
        #     obstacleY.append(i)
        #
        # for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
        #     obstacleY.append(car1[1, 3])
        #     obstacleX.append(i)


    # parked cars 2
    p_x = x_min + 2 * l_w + 2 * p_l + p_l / 2 + Car_obj.length / 2 - Car_obj.axleToBack
    p_yaw = np.pi
    j_indices = [0, 2, 3]

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
            line = np.linspace(car1[:, i], car1[:, i+1])
            obstacleX = obstacleX + line[:, 0].tolist()
            obstacleY = obstacleY + line[:, 1].tolist()

        # for i in range(car1.shape[1]-1):
        #     obstacleX.append(car1[0, i])
        #     obstacleY.append(car1[1, i])
        #
        # for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
        #     obstacleX.append(car1[0, 0])
        #     obstacleY.append(i)
        #
        # for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
        #     obstacleY.append(car1[1, 1])
        #     obstacleX.append(i)
        #
        # for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
        #     obstacleX.append(car1[0, 2])
        #     obstacleY.append(i)
        #
        # for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
        #     obstacleY.append(car1[1, 3])
        #     obstacleX.append(i)


    # parked cars 3
    p_x = x_min + 2 * l_w + 3 * p_l + p_l / 2 - (Car_obj.length / 2 - Car_obj.axleToBack)
    p_yaw = 0.0
    j_indices = [0, 1, 2, 3]

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
        for i in range(car1.shape[1] - 1):
            line = np.linspace(car1[:, i], car1[:, i + 1])
            obstacleX = obstacleX + line[:, 0].tolist()
            obstacleY = obstacleY + line[:, 1].tolist()

        # for i in range(car1.shape[1]-1):
        #     obstacleX.append(car1[0, i])
        #     obstacleY.append(car1[1, i])
        #
        # for i in range(math.floor(car1[1, 0]), math.ceil(car1[1, 1])-1, -1):
        #     obstacleX.append(car1[0, 0])
        #     obstacleY.append(i)
        #
        # for i in range(math.ceil(car1[0, 1]), math.floor(car1[0, 2])+1):
        #     obstacleY.append(car1[1, 1])
        #     obstacleX.append(i)
        #
        # for i in range(math.ceil(car1[1, 2]), math.floor(car1[1, 3])+1):
        #     obstacleX.append(car1[0, 2])
        #     obstacleY.append(i)
        #
        # for i in range(math.floor(car1[0, 3]), math.ceil(car1[0, 4]), -1):
        #     obstacleY.append(car1[1, 3])
        #     obstacleX.append(i)


    # for ax in axes:
    #     ax.scatter(parked_cars_x + moving_cars_x, parked_cars_y + moving_cars_y, color='black',
    #             marker='s', s=6)

    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_min)
        obstacleY.append(i)
    #
    for i in range(x_min, x_max+1):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in range(y_min, y_max+1):
        obstacleX.append(x_max)
        obstacleY.append(i)

    obstacleX.append(0.0)
    obstacleY.append(0.0)

    return x_min, x_max, y_min, y_max, p_w, p_l, l_w, n_r, n_s, n_s1, obstacleX, obstacleY, s

def evaluate_path(path_n):
    vel = np.diff(path_n[:, :2], axis=0)/MOTION_RESOLUTION # v_x, v_y
    acc = np.diff(vel[:, :2], axis=0)/MOTION_RESOLUTION

    motion_direction = np.arctan2(vel[:, 1], vel[:, 0])
    diff_yaw_motion_direction = np.cos(np.abs(path_n[:-1, 2] - motion_direction))
    reverse = diff_yaw_motion_direction < 0
    reverse_indices, = np.where(reverse)
    switchback_indices, = np.where(np.diff(reverse)==True)

    cost = np.sum(np.linalg.norm(acc, axis=1)) + BACK_COST*len(reverse_indices) + SB_COST*len(switchback_indices) + STEER_COST*(np.sum(np.abs(np.diff(path_n[:, 2], axis=0))%(2*np.pi))/MOTION_RESOLUTION)

    return cost

def parallel_run(start, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(hybrid_a_star_planning, [(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION) for goal in g_list])
    return results

def parallel_cost(path_list):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        costs = pool.starmap(evaluate_path, [(path_1) for path_1 in path_list])
    return costs

def parallel_ray(start, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list):
    futures = [hybrid_a_star_planning.remote(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION) for goal in g_list]
    results = ray.get(futures)
    return results

def transform_to_local(points, origin):
    """
    Transforms a list of points to a local frame defined by an origin and orientation angle.

    :param points: List of (x, y) points in the global frame
    :param origin: (x, y, theta) of the local frame origin in the global frame
    :return: Transformed list of points in the local frame
    """
    # Compute rotation matrix (2D)
    angle = origin[2]
    R = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])

    # Apply transformation
    local_points = (R @ (points - origin[:2]).T).T  # Rotate after translation

    return local_points

def transform_to_global(points, origin):
    # Compute rotation matrix (2D)
    angle = origin[2]
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])

    # Apply transformation
    global_points = (R @ (points).T).T + origin[:2]  # Rotate after translation

    return global_points

def points_in_polygon(points):
    polygon_vertices = [(-long_back, -lateral_bound), (long_front, -lateral_bound), (long_front, lateral_bound), (-long_back, lateral_bound)]
    polygon = Polygon(polygon_vertices)  # Create polygon
    truth_indices  = np.where([polygon.contains(Point(points[i, :])) for i in range(points.shape[0])])
    return points[truth_indices]

type='lot'

home_path = os.path.abspath(os.getcwd())

## Load the configuration files
config_path = home_path + '/Config/'

with open(config_path + 'config_planner.json') as f:
    config_planner = json.load(f)

with open(config_path + 'config_map.json') as f:
    config_map = json.load(f)

Car_obj = Car_class(config_planner)
x_min, x_max, y_min, y_max, p_w, p_l, l_w, n_r, n_s, n_s1, obstacleX, obstacleY, s = map_lot(type, config_map, Car_obj)

start = np.array(s)
obstacles = np.array([obstacleX, obstacleY]).T
obstacles_local = transform_to_local(obstacles, start)
obstacles_FOV_local = points_in_polygon(obstacles_local)
obstacles_FOV_local = np.vstack((obstacles_FOV_local, rect_points))
obstacles_FOV_global = transform_to_global(obstacles_FOV_local, start)

fig, ax = plt.subplots()

ax.plot(obstacles[:, 0], obstacles[:, 1], ".k")
ax.plot(obstacles_FOV_global[:, 0], obstacles_FOV_global[:, 1], ".r")
wb_2 = Car_obj.wheelBase/2
start = np.array([start[0] + wb_2*np.cos(start[2]),  s[1] + wb_2*np.sin(start[2]), start[2]])
plot_car(start[0], start[1], start[2], ax)
plt.show()
# ox = obstacleX
# oy = obstacleY

# Set Initial parameters
# start = [-5.0, 4.35, 0]
# goal = [0.0, 0.0, np.deg2rad(-90.0)]
# goal = [0, 0, np.deg2rad(-90.0)]
# goal = [0, 0, np.deg2rad(90)]
park_spots = [5, 9]
# park_spots_xy: list of centers of park_spots [x, y, 1 if left to center line and 0 if right to center line]
park_spots_xy = [np.array([x_min + (1 + (i // n_s)) * l_w + (i // n_s1) * p_l + p_l / 2,
                           y_min + l_w + (i % n_s1) * p_w + p_w / 2, bool(i % n_s <= n_s1 - 1)])
                 for i in park_spots]
goal_park_spots = []  # park_spot i ->  goal yaw = 0.0 if 0, and np.pi if 1 -> (x, y, yaw) of goal
for spot_xy in park_spots_xy:
    goal1 = np.array([spot_xy[0] - Car_obj.length / 2 + Car_obj.axleToBack, spot_xy[1], 0.0])
    goal2 = np.array([spot_xy[0] + Car_obj.length / 2 - Car_obj.axleToBack, spot_xy[1], np.pi])
    goal_spot_xy = [goal1, goal2]
    goal_plot = goal1
    goal_park_spots.append(goal_spot_xy)

goal_park_spots = list(chain.from_iterable(goal_park_spots))
g_list = [[g[0] + wb_2*np.cos(g[2]),  g[1] + wb_2*np.sin(g[2]), g[2]] for g in goal_park_spots]
# for _ in range(5):
#     g_list = g_list + g_list

def with_multiprocessing(start, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION):
    print("Starting function with multiprocessing.")
    jobs = []

    NUMBER_OF_PROCESSES = len(g_list)

    for i in range(NUMBER_OF_PROCESSES):
        process = multiprocessing.Process(
            target=hybrid_a_star_planning,
            args=(start, g_list[i], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION,)
        )
        jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     results = pool.starmap(hybrid_a_star_planning, [(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION) for goal in g_list])
    # return results

if __name__ == '__main__':

    ## Checking parallel computation
    # for i in range(3):
    #     g_list = g_list + g_list
    print("Start Hybrid A* planning")
    print("start : ", start)

    start_t_p = time.time()

    results = parallel_run(start, obstacles_FOV_global[:, 0].tolist(), obstacles_FOV_global[:, 1].tolist(), XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list)

    print("Comp time (parallelized): ", time.time() - start_t_p)

    path_list = [[np.array([path.x_list, path.y_list, path.yaw_list]).T] for path in results]
    # cost_2 = evaluate_path(path_list[-2][0])
    # evaluate_path(path)
    start_t_c = time.time()
    costs = parallel_cost(path_list)
    print("Comp time cost (parallelized): ", time.time() - start_t_p)

    start_t_m = time.time()
    with_multiprocessing(start, obstacles_FOV_global[:, 0].tolist(), obstacles_FOV_global[:, 1].tolist(), XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
    print("Comp time (multiprocess): ", time.time() - start_t_m)

    start_t = time.time()
    for goal in g_list:
        path = hybrid_a_star_planning(start, goal, obstacles_FOV_global[:, 0].tolist(), obstacles_FOV_global[:, 1].tolist(), XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
    print("Comp time: ", time.time() - start_t)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    dynamic_veh_0 = [
        np.array([x_min + l_w + 2 * p_l + l_w / 4, y_min + l_w + n_s1 * p_w, np.deg2rad(-90.0)])]
    dynamic_veh_vel = [np.array([0.0, -1.0])]

    ped_0 = [np.array([x_min + l_w + 2 * p_l, y_min + 0.75 * l_w])]
    ped_vel = [np.array([0.0, 0.5])]

    # car = np.array(
    #     [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
    #      [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    # rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
    #                       [math.sin(p_yaw), math.cos(p_yaw)]])
    # car = np.dot(rotationZ, car)
    # car1 = car + np.array([[p_x], [p_y]])

    circle = Circle((ped_0[0][0], ped_0[0][1]), radius=0.7, facecolor='red')
    p_yaw = dynamic_veh_0[0][2]
    p_x = dynamic_veh_0[0][0]
    p_y = dynamic_veh_0[0][1]
    i_x, i_y, i_yaw = start[0], start[1], start[2]
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            # axi = axes[i, j]
            plot_other_car(p_x, p_y, p_yaw, axes[i, j])
            circle = Circle((ped_0[0][0], ped_0[0][1]), radius=0.7, facecolor='red')
            # axes[i, j].plot(circle)
            axes[i, j].add_artist(circle)
            axes[i, j].plot(obstacles_FOV_global[:, 0].tolist(), obstacles_FOV_global[:, 1].tolist(), ".k")
            plot_car(i_x, i_y, i_yaw, axes[i, j])
            # axes[i, j].axis("equal")

    # Define how many colors you want
    num_colors = len(results)

    # for reeds_shepp curves
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    for i in range(len(results)):
        path = results[i]
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list
        label_i = 'Path Cost = ' + str(f"{costs[i]:.3f}")
        axes[i//2, i%2].plot(x, y, color=colors[i], label=label_i)
        for j in range(len(x)):
            plot_car_trans(x[j], y[j], yaw[j], axes[i//2, i%2])
        axes[i // 2, i % 2].legend(fontsize=20)

    # ax.grid(True)
    # start_t_r = time.time()
    #
    # results_r = parallel_ray(start, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list)
    #
    # print("Comp time (parallelized_ray): ", time.time() - start_t_r)


    # plt.legend()
    plt.show()
    # path = hybrid_a_star_planning(
    #     start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)