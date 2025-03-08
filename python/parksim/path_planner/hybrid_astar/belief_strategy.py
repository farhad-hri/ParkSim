import numpy as np
import os
import json
from itertools import chain
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle, Circle
import copy

from parksim.path_planner.hybrid_astar.hybrid_a_star import MOTION_RESOLUTION
from parksim.path_planner.hybrid_astar.hybrid_a_star_parallel import map_lot, Car_class, hybrid_a_star_planning, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, MOTION_RESOLUTION, PED_RAD
from parksim.path_planner.hybrid_astar.car import plot_car, plot_other_car
from parksim.path_planner.hybrid_astar.belief_pred_utils import occupancy_probability_multiple_spots_occ_dep

def plot_ell(p, Car_obj, ax):
    beta_l = 14
    beta_w = 17
    E = np.array([[beta_l * Car_obj.length, 0.0],
                  [0.0, beta_w * Car_obj.width]])

    a = np.sqrt(2 * (E[0, 0]))  # radius on the x-axis
    b = np.sqrt(2 * (E[1, 1]))  # radius on the y-axis
    angles = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    p_yaw = p[2]
    d = 1.0*Car_obj.length
    p_x = p[0] + d * np.cos(p_yaw)
    p_y = p[1] + d * np.sin(p_yaw)
    # Rotation matrix
    R = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                  [np.sin(p_yaw), np.cos(p_yaw)]])

    # Apply rotation
    rotated_points = R @ np.vstack((x, y))

    # Shift to center
    x_rotated = rotated_points[0, :] + p_x
    y_rotated = rotated_points[1, :] + p_y

    ax.plot(x_rotated, y_rotated, color='red', alpha=0.5)
    ax.plot(p_x, p_y, linestyle='', marker='o', color='red')  # center of ellipse

def rect_dist_obs_spots_plot(p, center_spots, Car_obj, ax):
    d = Car_obj.length
    p_x = p[0] + d * np.cos(p[2])
    p_y = p[1] + d * np.sin(p[2])
    p_yaw = p[2]

    beta_l = 3.0
    beta_w = 10.0

    spot_t = center_spots - np.array([p_x, p_y])
    rotationZ = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                          [np.sin(p_yaw), np.cos(p_yaw)]])
    spot_t = np.dot(spot_t, rotationZ)
    dist_inf_norm = np.max(
        np.array([1 / (beta_l * Car_obj.length / 2), 1 / (beta_w * Car_obj.width / 2)]) * np.abs(spot_t), axis=1)  # scale the infinity norm by length and width
    observed_spots = np.where(dist_inf_norm <= 1)[0]

    ## plot the 1 distance rectangle
    car = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    car[0, :] = beta_l * car[0, :]
    car[1, :] = beta_w * car[1, :]
    car = np.dot(rotationZ, car) # car is 2xN

    car1 = car + np.array([[p_x], [p_y]])  # (2xN) N are vertices
    ax.plot(car1[0, :], car1[1, :], color='blue', alpha=0.5)
    ax.plot(p_x, p_y, linestyle='', marker='o', color='blue')

    return observed_spots

def get_occ_vac_spots(static_obs_kd_tree, dynamic_veh_state, ped_points, center_spots, observed_spots, Car_obj, p_l, p_w):
    """

    Parameters:
    - ped_points: pedestrian points at current time step (n_ped, 2)
    - dynamic_veh_state: dynamic vehicle points at current time step (n_veh, 3)
    - observed_spots: indices (M,)

    Returns:
    - occ_spots ,vac_spots: Indices
    """

    ## For static obstacles
    epsilon_veh = 0.1
    dist_nearest_obst = static_obs_kd_tree.query(center_spots[observed_spots], p = np.inf,
                                                 distance_upper_bound=Car_obj.length / 2 + epsilon_veh)[0]  # 0 is distance to nearest neighbor of each spot, 1 is corresponding obstacle index
    observed_spots_occ_ind = np.where(dist_nearest_obst <= Car_obj.width / 2 + epsilon_veh)[0]

    ## For dynamic vehicles
    dynamic_veh_yaw = dynamic_veh_state[:, 2]
    car = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2]])
    rotationZ = np.array([[np.cos(dynamic_veh_yaw), -np.sin(dynamic_veh_yaw)],
                          [np.sin(dynamic_veh_yaw), np.cos(dynamic_veh_yaw)]]).transpose(2, 0, 1) # (n_veh, 2, 2)
    car = np.dot(rotationZ, car) # ( n_veh x 2(x, y) x 4(vertices) )
    car1 = car + dynamic_veh_state[:, :2, None] # (n_veh x 2(x, y) x 4)vertices))

    translated_x = car1[:, 0, :, None] - center_spots[observed_spots, 0]  # Shape (n_veh, 4, M)
    translated_y = car1[:, 1, :, None] - center_spots[observed_spots, 1]  # Shape (n_veh, 4, M)

    inside_x = (-p_l/2-epsilon_veh <= translated_x) & (translated_x <= p_l/2+epsilon_veh)  # Shape (n_veh, 4, M)
    inside_y = (-p_w/2-epsilon_veh <= translated_y) & (translated_y <= p_w/2+epsilon_veh)  # Shape (n_veh, 4, M)
    inside = inside_x & inside_y
    observed_spots_occ_ind_veh = np.where(np.any(inside, axis=(0, 1)))[0]

    ## For pedestrians
    epsilon_ped = 0.1
    translated_x = ped_points[:, 0, None] - center_spots[observed_spots, 0]  # Shape (n_ped, M)
    translated_y = ped_points[:, 1, None] - center_spots[observed_spots, 1]  # Shape (n_ped, M)

    inside_x = (-p_l/2-epsilon_ped <= translated_x) & (translated_x <= p_l/2+epsilon_ped)  # Shape (n_ped, M)
    inside_y = (-p_w/2-epsilon_ped <= translated_y) & (translated_y <= p_w/2+epsilon_ped)  # Shape (n_ped, M)
    inside = inside_x & inside_y
    observed_spots_occ_ind_ped = np.where(np.any(inside, axis=0))[0]

    occ_spots_ind = np.hstack((observed_spots_occ_ind, observed_spots_occ_ind_veh, observed_spots_occ_ind_ped))

    occ_spots_veh = observed_spots[observed_spots_occ_ind_veh]
    occ_spots_ped = observed_spots[observed_spots_occ_ind_ped]
    occ_spots_veh_only = sorted(list(set(occ_spots_veh) - set(observed_spots[observed_spots_occ_ind]))) # subtract occupied spots by static obstacles

    ped_minus_veh = set(occ_spots_ped) - set(occ_spots_veh) # subtract occupied spots by vehicle from pedestrians
    occ_spots_ped_only = sorted(list(ped_minus_veh - set(observed_spots[observed_spots_occ_ind]))) # subtract occupied spots by static obstacles

    occ_spots = sorted(set(list(observed_spots[observed_spots_occ_ind]))) # occupied only by static obstacles
    occ_spots_all = sorted(set(list(observed_spots[occ_spots_ind]))) # occupied by both static and dynamic agents
    vac_spots = sorted(list(set(observed_spots) - set(occ_spots_all)))

    return occ_spots, vac_spots, occ_spots_veh_only, occ_spots_ped_only

def get_vertices_car(Car_obj, p):
    p_x = p[0]
    p_y = p[1]
    p_yaw = p[2]
    car = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    car[0, :] = 1 * car[0, :]
    car[1, :] = 1 * car[1, :]
    rotationZ = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                          [np.sin(p_yaw), np.cos(p_yaw)]])
    car = np.dot(rotationZ, car) # car is 2xN
    car1 = car + np.array([[p_x], [p_y]])  # (2xN) N are vertices

    return car1[0, :].tolist(), car1[1, :].tolist()

type='big_lot'

home_path = os.path.abspath(os.getcwd())

## Load the configuration files
config_path = home_path + '/Config/'

with open(config_path + 'config_planner.json') as f:
    config_planner = json.load(f)

with open(config_path + 'config_map.json') as f:
    config_map = json.load(f)

Car_obj = Car_class(config_planner)
fig, ax = plt.subplots(figsize=(10, 8))
x_min, x_max, y_min, y_max, p_w, p_l, l_w, n_r, n_s, n_s1, obstacleX, obstacleY, s, center_spots, occ_spot_indices = map_lot(type, config_map, Car_obj, ax)

p = np.array(s)

## Get the complete path of dynamic agents
# Vehicle
# dynamic_veh_0 = np.array([np.hstack((center_spots[29] + np.array([Car_obj.length/4, 0.0]), np.deg2rad(0.0))),
#                           np.hstack((center_spots[32] + np.array([-Car_obj.length/2-l_w/4, -l_w/2]), np.deg2rad(90.0)))
#                           ])

# dynamic_veh_goal = np.array([np.hstack((center_spots[29] + np.array([-Car_obj.length/2-l_w/4, 2.0]), np.deg2rad(90.0))),
#                              np.hstack((center_spots[35], np.deg2rad(0.0)))
#                           ])

dynamic_veh_0 = np.array([np.hstack((center_spots[35] + np.array([-Car_obj.length/2 - l_w/3, p_w]), np.deg2rad(110.0))),
                          np.hstack((center_spots[32] + np.array([-Car_obj.length/2-l_w/4, -l_w/2]), np.deg2rad(90.0)))
                          ])

dynamic_veh_goal = np.array([np.hstack((center_spots[29], np.deg2rad(-180.0))),
                             np.hstack((center_spots[35], np.deg2rad(0.0)))
                          ])

dynamic_veh_parking = [1, 1]
T = 30 # total number of time steps
length_preds = T+1
dynamic_veh_path = []
ego_obst_x, ego_obst_y = get_vertices_car(Car_obj, p)
obstacleX_dyn = obstacleX + ego_obst_x
obstacleY_dyn = obstacleY + ego_obst_y
for veh_i, veh_parking in enumerate(dynamic_veh_parking):
    if veh_parking:
        ## dynamic_veh out of spot
        path_veh = hybrid_a_star_planning(dynamic_veh_0[veh_i], dynamic_veh_goal[veh_i], obstacleX_dyn, obstacleY_dyn, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        if length_preds > len(path_veh.x_list):
            extra_time_steps = int(length_preds - len(path_veh.x_list))
            path_veh_n = np.array([path_veh.x_list, path_veh.y_list, path_veh.yaw_list]).T
            last_state = path_veh_n[-1]
            repeat_veh = np.repeat(last_state.reshape((1, -1)), repeats=extra_time_steps, axis=0)
            path_veh_n = np.vstack((path_veh_n, repeat_veh))
        else:
            path_veh_n = np.array([path_veh.x_list, path_veh.y_list, path_veh.yaw_list]).T
            path_veh_n = path_veh_n[:length_preds]

    dynamic_veh_path.append(path_veh_n)

dynamic_veh_path=np.array(dynamic_veh_path)

# Pedestrian
ped_0 = np.array([center_spots[28] + np.array([Car_obj.length/2-1, -Car_obj.width/2-1])
                  ])
ped_vel = np.array([[-0.9, -1.0]
                    ])
# time steps
delay = 10 # should be less than or equal to T, length_preds = T+1
moving_time = int(length_preds - delay)

ped_init = np.array([np.repeat(np.array([ped_0[j]]), repeats = delay, axis=0) for j in range(ped_0.shape[0])])
ped_path = np.array([np.vstack((ped_init[j], np.array([ped_0[j] + MOTION_RESOLUTION*i*ped_vel[j] for i in range(moving_time)]))) for j in range(len(ped_0))])

## construct KDTree for static obstacles
static_xy = np.array([obstacleX, obstacleY]).T
static_obs_kd_tree = cKDTree(static_xy)

t=10
T_pred = 5 # time steps
dynamic_veh_path_t = dynamic_veh_path[:, t:t+T_pred+1]
ped_path_t = ped_path[:, t:t+T_pred+1]

p = np.array(s)

# plot_ell(s, Car_obj, ax)
observed_spots = rect_dist_obs_spots_plot(p, center_spots, Car_obj, ax)
print("Observed Spots: ", observed_spots)

occ_spots, vac_spots, occ_spots_veh, occ_spots_ped = get_occ_vac_spots(static_obs_kd_tree, dynamic_veh_path_t[:, 0, :], ped_path_t[:, 0, :], center_spots, observed_spots, Car_obj, p_l, p_w)
print(f"Occupied spots: {occ_spots}, Vacant Spots: {vac_spots}")
print(f"Occupied by Vehicle: {occ_spots_veh}, Occupied by Pedestrian: {occ_spots_ped}")

Sigma_0 = np.repeat(np.array([[[0.5*Car_obj.length, 0], [0, 0.5*Car_obj.width]]]), repeats=dynamic_veh_0.shape[0], axis=0)  # Covariance for each vehicle
Sigma_0_ped = np.repeat(np.array([[[PED_RAD, 0], [0, PED_RAD]]]), repeats=ped_0.shape[0], axis=0) # Covariance for each pedestrian
Q = np.array([[0.5, 0], [0, 0.5]])  # Process noise (uncertainty growth)

P_O_vacant, P_O_occ = occupancy_probability_multiple_spots_occ_dep(T_pred, dynamic_veh_path_t, ped_path_t, Sigma_0, Sigma_0_ped, Q, center_spots, vac_spots, occ_spots_veh, occ_spots_ped, Car_obj)

# Plotting the dynamic agents
for time_dynamic in range(T_pred+1):
    # plot vehicle
    for veh_i in range(dynamic_veh_path_t.shape[0]):
        p_yaw = dynamic_veh_path_t[veh_i, time_dynamic, 2]
        p_x = dynamic_veh_path_t[veh_i, time_dynamic, 0]
        p_y = dynamic_veh_path_t[veh_i, time_dynamic, 1]
        plot_other_car(p_x, p_y, p_yaw, ax)

    # plot pedestrians
    for ped_i in range(ped_path_t.shape[0]):
        circle = Circle((ped_path_t[ped_i, time_dynamic, 0], ped_path_t[ped_i, time_dynamic, 1]), radius=PED_RAD,
                        facecolor='red', alpha=0.5)
        ax.add_artist(circle)

ax.axis('equal')
plt.show()