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
import multiprocessing
from scipy.spatial.transform import Rotation as Rot
from matplotlib.animation import FuncAnimation, FFMpegWriter

from parksim.path_planner.hybrid_astar.hybrid_a_star import MOTION_RESOLUTION
from parksim.path_planner.hybrid_astar.hybrid_a_star_parallel import map_lot, Car_class, evaluate_path, hybrid_a_star_planning, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, MOTION_RESOLUTION, PED_RAD, MAX_WAIT_TIME
from parksim.path_planner.hybrid_astar.car import plot_car, plot_other_car, plot_other_car_return, plot_car_return, VRX, VRY, plot_other_car_trans, plot_car_trans
from parksim.path_planner.hybrid_astar.belief_pred_utils import occupancy_probability_multiple_spots_occ_dep ,occupancy_probability_multiple_spots_occ_dep_p


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

def valid_point(p, map_limits):
    x_min, x_max, y_min, y_max = map_limits
    if p[0] <= x_min or p[0] >= x_max or p[1] <= y_min or p[1] >= y_max:
        return False
    return True

def spline_inter(start, goal):
    x0, y0, theta0 = start[0], start[1], start[2]
    x_f, y_f, theta_f = goal[0], goal[1], goal[2]
    # Compute derivatives from headings
    vx0, vy0 = np.cos(theta0), np.sin(theta0)
    vx_f, vy_f = np.cos(theta_f), np.sin(theta_f)

    # Set up equations for quintic polynomial: a*t^5 + b*t^4 + c*t^3 + d*t^2 + e*t + f
    A = np.array([
        [0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0],
        [5, 4, 3, 2, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [20, 12, 6, 2, 0, 0]
    ])

    bx = np.array([x0, x_f, vx0, vx_f, 0, 0])
    by = np.array([y0, y_f, vy0, vy_f, 0, 0])

    # Solve for coefficients
    coeff_x = np.linalg.solve(A, bx)
    coeff_y = np.linalg.solve(A, by)

    # Generate the spline trajectory
    T = np.linspace(0, 1, 10) # 10 is number of spline points
    X = sum(c * T**i for i, c in enumerate(reversed(coeff_x)))
    Y = sum(c * T**i for i, c in enumerate(reversed(coeff_y)))

    # Compute derivatives for heading
    dX = sum(i * c * T**(i-1) for i, c in enumerate(reversed(coeff_x)) if i > 0)
    dY = sum(i * c * T**(i-1) for i, c in enumerate(reversed(coeff_y)) if i > 0)


    # Compute heading at each time step
    headings = np.arctan2(dY, dX)

    return X, Y, headings

def rect_dist_obs_spots_plot(p, center_spots, roads, roads_y, map_limits, Car_obj, ax):
    d = Car_obj.length
    p_x = p[0] + d * np.cos(p[2])
    p_y = p[1] + d * np.sin(p[2])
    p_yaw = p[2]

    beta_l = 3.0
    beta_w = 10.0

    spot_t = center_spots - np.array([p_x, p_y]) # translate the frame (wrt ego's center)
    rotationZ = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                          [np.sin(p_yaw), np.cos(p_yaw)]])
    spot_t = np.dot(spot_t, rotationZ) # rotate the frame (wrt ego's heading)

    dist_inf_norm = np.max(
        np.array([1 / (beta_l * Car_obj.length / 2), 1 / (beta_w * Car_obj.width / 2)]) * np.abs(spot_t), axis=1)  # scale the infinity norm by length and width
    observed_spots = np.where(dist_inf_norm <= 1)[0]
    
    ## For exploration
    # only x is relevant in roads_ego

    # vertical lane
    if abs(abs(p_yaw/(np.pi/2))-1) < 0.0001:
        roads = np.array([[0]*len(roads_y), roads_y]).T
    # horizontal lane
    elif abs(abs(p_yaw/(np.pi)) - 0)<=0.0001 or abs(abs(p_yaw/(np.pi)) - 1)<=0.0001:
        roads = np.array([roads_x, [0]*len(roads_x)]).T

    # wrt ego's center
    x_max = (beta_l - 0.5)*Car_obj.length
    y_max = (beta_w)*Car_obj.width/2
    y_min = -(beta_w)*Car_obj.width/2
    roads_ego = roads - np.array([p[0], p[1]])
    roads_ego = np.dot(roads_ego, rotationZ)
    roads_in_FOV = roads_ego[np.where(roads_ego[:, 0] >= 0)[0], 0]
    roads_in_FOV = roads_in_FOV[np.where(roads_in_FOV < x_max)[0]]

    straight = [x_max, 0.0, p_yaw]
    if roads_in_FOV.size > 0:
        x_center = roads_in_FOV[0] # change this depending on center of new road
        # angle wrap is (x + pi)%(2*pi) - pi for angles in [-pi, pi]
        # p_yaw + pi/2 for left, p_yaw - pi/2 for right assuming p_yaw in {0, pi/2, pi, -pi/2}
        left = [x_center + l_w/4, y_max, (p_yaw + np.pi/2 + np.pi) % (2 * np.pi) - np.pi]
        right = [x_center - l_w/4, y_min, (p_yaw - np.pi/2 + np.pi) % (2 * np.pi) - np.pi]
        explore_points = np.array([straight, left, right])
    else:
        explore_points = np.array([straight])
    
    explore_points[:, :2] = np.dot(explore_points[:, :2], rotationZ.T) # rotate the frame (from ego's heading to global frame)
    explore_points[:, :2] = explore_points[:, :2] + np.array([p[0], p[1]]) # translate the frame (from ego's center to global frame)

    explore_points_final = np.array([explore_points[i] for i in range(explore_points.shape[0]) 
                                     if valid_point(explore_points[i], map_limits)])

    ## plot the 1 distance rectangle
    car = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    car[0, :] = beta_l * car[0, :]
    car[1, :] = beta_w * car[1, :]
    car = np.dot(rotationZ, car) # car is 2xN

    car1 = car + np.array([[p_x], [p_y]])  # (2xN) N are vertices
    ax.plot(car1[0, :], car1[1, :], color='blue', alpha=0.2)
    ax.plot(p_x, p_y, linestyle='', marker='o', color='blue')
    ax.plot(explore_points_final[:, 0], explore_points_final[:, 1], linestyle='', marker='o', color='green')    

    return observed_spots, explore_points_final

def rect_dist_obs_p_spots_plot(p, center_spots, roads, roads_y, map_limits, Car_obj, ax):

    beta_l = 3.0
    beta_w = 10.0
    long_scale = beta_l * Car_obj.length
    lat_scale = beta_w * Car_obj.width

    d = (beta_l - 1)* (Car_obj.length / 2) 
    p_x = p[0] + d * np.cos(p[2])
    p_y = p[1] + d * np.sin(p[2])
    p_yaw = p[2]

    spot_t = center_spots - np.array([p_x, p_y]) # translate the frame (wrt ego's center)
    rotationZ = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                          [np.sin(p_yaw), np.cos(p_yaw)]])
    spot_t = np.dot(spot_t, rotationZ) # rotate the frame (wrt ego's heading)

    epsilon = 1.0
    dist_inf_norm = np.max(
        np.array([1 / (long_scale / 2), 1 / (lat_scale / 2)]) * np.abs(spot_t), axis=1)  # scale the infinity norm by length and width
    observed_spots = np.where(dist_inf_norm <= epsilon)[0]

    # p is partially observed
    gamma = 2.0
    # d_ex = (beta_l*p_ex - 1)* (Car_obj.length / 2) 
    # p_ex_x = p[0] + d_ex * np.cos(p[2])
    # p_ex_y = p[1] + d_ex * np.sin(p[2])
    # p_yaw = p[2]

    # spot_ex_t = center_spots - np.array([p_x, p_y]) 
    observed_p_all_spots = np.where(dist_inf_norm <= gamma)[0]
    observed_p_spots = np.array(list(set(observed_p_all_spots) -  set(observed_spots)))

    # alpha = 0.1
    r_d = np.minimum((np.log(2)/(gamma - epsilon))*np.maximum((dist_inf_norm - epsilon), 0.0), np.log(2))  # linear r(d) function, can be softmax like
    p_c = np.exp(-r_d) # probability of correct observation for partially observed spots   

    ## For exploration
    # only x is relevant in roads_ego

    # vertical lane
    if abs(abs(p_yaw/(np.pi/2))-1) < 0.0001:
        roads = np.array([[0]*len(roads_y), roads_y]).T
    # horizontal lane
    elif abs(abs(p_yaw/(np.pi)) - 0)<=0.0001 or abs(abs(p_yaw/(np.pi)) - 1)<=0.0001:
        roads = np.array([roads_x, [0]*len(roads_x)]).T

    # wrt ego's center
    x_max = (beta_l - 0.5)*Car_obj.length
    y_max = (beta_w)*Car_obj.width/2
    y_min = -(beta_w)*Car_obj.width/2
    roads_ego = roads - np.array([p[0], p[1]])
    roads_ego = np.dot(roads_ego, rotationZ)
    roads_in_FOV = roads_ego[np.where(roads_ego[:, 0] >= 0)[0], 0]
    roads_in_FOV = roads_in_FOV[np.where(roads_in_FOV < x_max)[0]]

    straight = [x_max - Car_obj.length, 0.0, p_yaw]
    x_max_global = np.dot(np.array([[x_max, 0.0]]), rotationZ.T) + np.array([p[0], p[1]])
    explore_points_l = []
    if valid_point(x_max_global[0], map_limits):
        explore_points_l = explore_points_l + [straight]

    if roads_in_FOV.size > 0:
        x_center = roads_in_FOV[0] # change this depending on center of new road
        # angle wrap is (x + pi)%(2*pi) - pi for angles in [-pi, pi]
        # p_yaw + pi/2 for left, p_yaw - pi/2 for right assuming p_yaw in {0, pi/2, pi, -pi/2}
        left = [x_center + l_w/4, y_max, (p_yaw + np.pi/2 + np.pi) % (2 * np.pi) - np.pi]
        right = [x_center - l_w/4, y_min, (p_yaw - np.pi/2 + np.pi) % (2 * np.pi) - np.pi]
        explore_points_l = explore_points_l + [left] + [right]

    explore_points = np.array(explore_points_l)
    
    explore_points[:, :2] = np.dot(explore_points[:, :2], rotationZ.T) # rotate the frame (from ego's heading to global frame)
    explore_points[:, :2] = explore_points[:, :2] + np.array([p[0], p[1]]) # translate the frame (from ego's center to global frame)

    explore_points_final = np.array([explore_points[i] for i in range(explore_points.shape[0]) 
                                     if valid_point(explore_points[i], map_limits)])

    ## plot the 1 distance rectangle
    car = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    car[0, :] = beta_l * car[0, :]
    car[1, :] = beta_w * car[1, :]

    car_o = np.dot(rotationZ, np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])) # car is 2xN
    car1_o = car_o + np.array([[p[0]], [p[1]]])  # (2xN) N are vertices

    car = np.dot(rotationZ, car) # car is 2xN
    car1 = car + np.array([[p_x], [p_y]])  # (2xN) N are vertices
    ax.plot(car1[0, :], car1[1, :], color='blue', alpha=0.2)

    car_p = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    car_p[0, :] = (beta_l*gamma) * car_p[0, :]
    car_p[1, :] = (beta_w*gamma) * car_p[1, :]
    car_p = np.dot(rotationZ, car_p) # car is 2xN
    car1_p = car_p + np.array([[p_x], [p_y]])  # (2xN) N are vertices
    ax.plot(car1_p[0, :], car1_p[1, :], color='orange', alpha=0.2)
    # ax.plot(p_x, p_y, linestyle='', marker='o', color='blue')    
    ax.plot(car1_o[0, :], car1_o[1, :], color='green')

    ax.plot(explore_points_final[:, 0], explore_points_final[:, 1], linestyle='', marker='o', color='green')    

    return observed_spots, explore_points_final, observed_p_spots, dist_inf_norm, p_c

def rect_dist_obs_p_spots_e(p, center_spots, roads, roads_y, map_limits, Car_obj):

    beta_l = 3.0
    beta_w = 10.0
    long_scale = beta_l * Car_obj.length
    lat_scale = beta_w * Car_obj.width

    d = (beta_l - 1)* (Car_obj.length / 2) 
    p_x = p[0] + d * np.cos(p[2])
    p_y = p[1] + d * np.sin(p[2])
    p_yaw = p[2]

    spot_t = center_spots - np.array([p_x, p_y]) # translate the frame (wrt ego's center)
    rotationZ = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                          [np.sin(p_yaw), np.cos(p_yaw)]])
    spot_t = np.dot(spot_t, rotationZ) # rotate the frame (wrt ego's heading)

    epsilon = 1.0
    dist_inf_norm = np.max(
        np.array([1 / (long_scale / 2), 1 / (lat_scale / 2)]) * np.abs(spot_t), axis=1)  # scale the infinity norm by length and width
    observed_spots = np.where(dist_inf_norm <= epsilon)[0]

    # p is partially observed
    gamma = 2.0
    # d_ex = (beta_l*p_ex - 1)* (Car_obj.length / 2) 
    # p_ex_x = p[0] + d_ex * np.cos(p[2])
    # p_ex_y = p[1] + d_ex * np.sin(p[2])
    # p_yaw = p[2]

    # spot_ex_t = center_spots - np.array([p_x, p_y]) 
    observed_p_all_spots = np.where(dist_inf_norm <= gamma)[0]
    observed_p_spots = np.array(list(set(observed_p_all_spots) -  set(observed_spots)))

    r_d = np.minimum((np.log(2)/(gamma - epsilon))*np.maximum((dist_inf_norm - epsilon), 0.0), np.log(2))  # linear r(d) function, can be softmax like
    p_c = np.exp(-r_d) # probability of correct observation for partially observed spots   

    ## For exploration
    # only x is relevant in roads_ego

    # vertical lane
    if abs(abs(p_yaw/(np.pi/2))-1) < 0.0001:
        roads = np.array([[0]*len(roads_y), roads_y]).T
    # horizontal lane
    elif abs(abs(p_yaw/(np.pi)) - 0)<=0.0001 or abs(abs(p_yaw/(np.pi)) - 1)<=0.0001:
        roads = np.array([roads_x, [0]*len(roads_x)]).T

    # wrt ego's center
    x_max = (beta_l - 0.5)*Car_obj.length
    y_max = (beta_w)*Car_obj.width/2
    y_min = -(beta_w)*Car_obj.width/2
    roads_ego = roads - np.array([p[0], p[1]])
    roads_ego = np.dot(roads_ego, rotationZ)
    roads_in_FOV = roads_ego[np.where(roads_ego[:, 0] >= 0)[0], 0]
    roads_in_FOV = roads_in_FOV[np.where(roads_in_FOV < x_max)[0]]

    straight = [x_max - Car_obj.length, 0.0, p_yaw]
    if roads_in_FOV.size > 0:
        x_center = roads_in_FOV[0] # change this depending on center of new road
        # angle wrap is (x + pi)%(2*pi) - pi for angles in [-pi, pi]
        # p_yaw + pi/2 for left, p_yaw - pi/2 for right assuming p_yaw in {0, pi/2, pi, -pi/2}
        left = [x_center + l_w/4, y_max, (p_yaw + np.pi/2 + np.pi) % (2 * np.pi) - np.pi]
        right = [x_center - l_w/4, y_min, (p_yaw - np.pi/2 + np.pi) % (2 * np.pi) - np.pi]
        explore_points = np.array([straight, left, right])
    else:
        explore_points = np.array([straight])
    
    explore_points[:, :2] = np.dot(explore_points[:, :2], rotationZ.T) # rotate the frame (from ego's heading to global frame)
    explore_points[:, :2] = explore_points[:, :2] + np.array([p[0], p[1]]) # translate the frame (from ego's center to global frame)

    explore_points_final = np.array([explore_points[i] for i in range(explore_points.shape[0]) 
                                     if valid_point(explore_points[i], map_limits)])

    return observed_spots, explore_points_final, observed_p_spots, dist_inf_norm, p_c


def update_spots_occupancy_arr_dep(P_O_spots, T):
    P_O_result = np.zeros((T + 1, len(P_O_spots)))
    P_O_result[0] = P_O_spots
    p_a = 0.001 # probability of arrival
    p_d = 0.0005 # probability of departure
    for k in range(1, T + 1):
        P_O_result[k] = p_a*(1-P_O_result[k-1]) + (1-p_d)*P_O_result[k-1]

    P_O_result = np.maximum(P_O_result, 0.00001)
    return P_O_result

def get_occ_vac_spots_stat(static_obs_kd_tree, center_spots, observed_spots, Car_obj, p_l, p_w):
    """

    Parameters:
    - observed_spots: indices (M,)

    Returns:
    - occ_spots ,vac_spots: Indices
    """

    ## For static obstacles
    epsilon_veh = 0.1
    dist_nearest_obst = static_obs_kd_tree.query(center_spots[observed_spots], p = np.inf,
                                                 distance_upper_bound=Car_obj.length / 2 + epsilon_veh)[0]  # 0 is distance to nearest neighbor of each spot, 1 is corresponding obstacle index
    observed_spots_occ_ind = np.where(dist_nearest_obst <= Car_obj.width / 2 + epsilon_veh)[0]

    occ_spots_ind = observed_spots_occ_ind

    occ_spots = sorted(set(list(observed_spots[observed_spots_occ_ind]))) # occupied only by static obstacles
    occ_spots_all = sorted(set(list(observed_spots[occ_spots_ind]))) # occupied by both static and dynamic agents
    vac_spots = sorted(list(set(observed_spots) - set(occ_spots_all)))    

    return occ_spots, vac_spots


## for fully occupied observations
def update_spots_occupancy_arr_dep_p_o(P_O_spots, p_c, T):
    P_O_result = np.zeros((T + 1, len(P_O_spots)))
    P_O_prior = np.zeros((T + 1, len(P_O_spots)))
    P_O_result[0] = P_O_spots
    p_a = 0.001 # probability of arrival
    p_d = 0.0005 # probability of departure
    for k in range(1, T + 1):
        P_O_prior[k] = p_a*(1-P_O_result[k-1]) + (1-p_d)*P_O_result[k-1]
        prior_k = p_c*P_O_prior[k] + (1-p_c)*(1-P_O_prior[k]) # for fully occupied spots
        P_O_result[k] = p_c*P_O_prior[k]/prior_k

    P_O_result = np.maximum(P_O_result, 0.0001)
    return P_O_result

## for fully vacant observations
def update_spots_occupancy_arr_dep_p_v(P_O_spots, p_c, T):
    P_O_result = np.zeros((T + 1, len(P_O_spots)))
    P_O_prior = np.zeros((T + 1, len(P_O_spots)))
    P_O_result[0] = P_O_spots
    p_a = 0.001 # probability of arrival
    p_d = 0.0005 # probability of departure
    for k in range(1, T + 1):
        P_O_prior[k] = p_a*(1-P_O_result[k-1]) + (1-p_d)*P_O_result[k-1]
        prior_k = (1-p_c)*P_O_prior[k] + (p_c)*(1-P_O_prior[k]) # for fully occupied spots
        P_O_result[k] = p_c*P_O_prior[k]/prior_k

    P_O_result = np.maximum(P_O_result, 0.00001)
    return P_O_result

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

    occ_spots_dyn = occ_spots_veh_only + occ_spots_ped_only

    return occ_spots, vac_spots, occ_spots_dyn, occ_spots_veh_only, occ_spots_ped_only

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

def get_points_car(Car_obj, p):
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

    obstacleX = []
    obstacleY = []
    ## Car
    for i in range(car1.shape[1] - 1):
        line = np.linspace(car1[:, i], car1[:, i + 1], num=10, endpoint=False)
        obstacleX = obstacleX + line[:, 0].tolist()
        obstacleY = obstacleY + line[:, 1].tolist()

    return obstacleX, obstacleY

def get_ped_points(PED_RAD, p):
    p_x = p[0]
    p_y = p[1]

    angles = np.linspace(0, 2*np.pi, num=10)

    obstacleX = []
    obstacleY = []
    ## Car
    for i in range(angles.shape[0]):
        obstacleX = obstacleX + [p_x + PED_RAD*np.cos(angles[i])]
        obstacleY = obstacleY + [p_y + PED_RAD*np.sin(angles[i])]

    return obstacleX, obstacleY

def parallel_run(start, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(hybrid_a_star_planning, [(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION) for goal in g_list])
    return results

def plot_anim(ax, fig, p_all, dynamic_veh_path, ped_path):

    # Define the update function for the animation
    extend_end = 5
    time_traj = np.arange(p_all.shape[0])
    p_all = np.vstack((p_all, [p_all[-1]]*extend_end))
    time_traj = np.hstack((time_traj, [time_traj[-1]]*extend_end))

    end_time = int(time_traj[-1])
    dynamic_veh_path = dynamic_veh_path[:, :end_time+1, :]
    ped_path = ped_path[:, :end_time+1, :]

    # Repeat the last index 5 times
    last_time_repeated_veh = np.repeat(dynamic_veh_path[:, -1:, :], extend_end, axis=1)
    # Concatenate along axis 1
    dynamic_veh_path = np.concatenate([dynamic_veh_path, last_time_repeated_veh], axis=1)

    # Repeat the last index 5 times
    last_time_repeated_ped = np.repeat(ped_path[:, -1:, :], extend_end, axis=1)
    # Concatenate along axis 1
    ped_path = np.concatenate([ped_path, last_time_repeated_ped], axis=1)

    dyn_cars = []
    dyn_cars_arrow = []
    for veh_i in range(dynamic_veh_path.shape[0]):
        p_yaw = dynamic_veh_path[veh_i, 0, 2]
        p_x = dynamic_veh_path[veh_i, 0, 0]
        p_y = dynamic_veh_path[veh_i, 0, 1]
        car_plot_d, arrow_plot_d = plot_other_car_return(p_x, p_y, p_yaw, ax)
        dyn_cars.append(car_plot_d)
        dyn_cars_arrow.append(arrow_plot_d)

    ped = []
    # plot pedestrians
    for ped_i in range(ped_path.shape[0]):
        circle = Circle((ped_path[ped_i, 0, 0], ped_path[ped_i, 0, 1]), radius=PED_RAD,
                        facecolor='red', alpha=0.5)
        ped.append(circle)
        ax.add_artist(circle)

    car_plot, arrow_plot = plot_car_return(p_all[0, 0], p_all[0, 1], p_all[0, 2], ax)

    beta_l = 3.0
    beta_w = 10.0
    d = (beta_l - 1)* (Car_obj.length / 2) 
    p_x_f = p[0] + d * np.cos(p[2])
    p_y_f = p[1] + d * np.sin(p[2])
    ## plot the 1 distance rectangle
    car_FOV_o = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    car_FOV_o[0, :] = beta_l * car_FOV_o[0, :]
    car_FOV_o[1, :] = beta_w * car_FOV_o[1, :]
    rotationZ = np.array([[np.cos(p_all[0, 2]), -np.sin(p_all[0, 2])],
                        [np.sin(p_all[0, 2]), np.cos(p_all[0, 2])]])
    car_FOV = np.dot(rotationZ, car_FOV_o) # car is 2xN
    car1_FOV = car_FOV + np.array([[p_x_f], [p_y_f]])  # (2xN) N are vertices
    car_FOV_plot, = ax.plot(car1_FOV[0, :], car1_FOV[1, :], color='blue', alpha=0.2)

    # expanding rectangle for partially observed spots
    gamma = 2.0 
    car_p_o = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])
    car_p_o[0, :] = gamma*beta_l*car_p_o[0, :]
    car_p_o[1, :] = gamma*beta_w*car_p_o[1, :]
    car_p = np.dot(rotationZ, car_p_o) # car is 2xN
    car1_p = car_p + np.array([[p_x_f], [p_y_f]])  # (2xN) N are vertices
    car_FOV_plot_p, = ax.plot(car1_p[0, :], car1_p[1, :], color='orange', alpha=0.2)

    time_text = 't=' + str(time_traj[0])
    props = dict(boxstyle='round', facecolor='w', alpha=0.5, edgecolor='black', linewidth=2)
    text_t  = ax.text(2.5,30.5, time_text, fontsize=22, bbox=props)

    def update(frame):

        for veh_i in range(dynamic_veh_path.shape[0]):
            p_yaw = dynamic_veh_path[veh_i, frame, 2]
            p_x = dynamic_veh_path[veh_i, frame, 0]
            p_y = dynamic_veh_path[veh_i, frame, 1]

            rot = Rot.from_euler('z', -p_yaw).as_matrix()[0:2, 0:2]
            car_outline_x, car_outline_y = [], []
            for rx, ry in zip(VRX, VRY):
                converted_xy = np.stack([rx, ry]).T @ rot
                car_outline_x.append(converted_xy[0] + p_x)
                car_outline_y.append(converted_xy[1] + p_y)

            dyn_cars[veh_i].set_data(car_outline_x, car_outline_y)
            dyn_cars_arrow[veh_i].xy = [p_x + 1 * np.cos(p_yaw), p_y + 1 * np.sin(p_yaw)]
            dyn_cars_arrow[veh_i].set_position((p_x, p_y))

        # plot pedestrians
        for ped_i in range(ped_path.shape[0]):
            ped[ped_i].center = (ped_path[ped_i, frame, 0], ped_path[ped_i, frame, 1],)

        p_yaw = p_all[frame, 2]
        p_x =  p_all[frame, 0]
        p_y =  p_all[frame, 1]

        rot = Rot.from_euler('z', -p_yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(VRX, VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0] + p_x)
            car_outline_y.append(converted_xy[1] + p_y)

        car_plot.set_data(car_outline_x, car_outline_y)
        arrow_plot.xy = [p_x + 1 * np.cos(p_yaw), p_y + 1 * np.sin(p_yaw)]
        arrow_plot.set_position((p_x, p_y))

        text_t.set_text('t=' + str(time_traj[frame]))

        p_x_f = p_x + d * np.cos(p_yaw)
        p_y_f = p_y + d * np.sin(p_yaw)
        rotationZ = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                        [np.sin(p_yaw), np.cos(p_yaw)]])
        car_FOV = np.dot(rotationZ, car_FOV_o) # car is 2xN
        car1_FOV = car_FOV + np.array([[p_x_f], [p_y_f]])  # (2xN) N are vertices
        car_FOV_plot.set_data(car1_FOV[0, :], car1_FOV[1, :])

        car_p = np.dot(rotationZ, car_p_o) # car is 2xN
        car1_p = car_p + np.array([[p_x_f], [p_y_f]])  # (2xN) N are vertices
        car_FOV_plot_p.set_data(car1_p[0, :], car1_p[1, :])

        # car = drawCar(Car_obj, x[frame], y[frame], yaw[frame])
        # car_plot_a.set_data(car[0, :], car[1, :])
        # arrow_plot_a.xy = [x[frame]+1*math.cos(yaw[frame]), y[frame]+1*math.sin(yaw[frame])]
        # arrow_plot_a.set_position((x[frame], y[frame]))
        # text_t.set_text('t=' + str(time_traj[frame]))
        # for i in range(len(dynamic_plot_a)):
        #     # Update circle positions
        #     dynamic_plot_a[i].center = (obst_x[frame][i], obst_y[frame][i],)
        # # dynamic_plot_a.set_data(obst_x[frame], obst_y[frame])

        plot_list = dyn_cars + dyn_cars_arrow + [car_plot] + [arrow_plot] + [text_t] + ped + [car_FOV_plot] + [car_FOV_plot_p]
        return plot_list

    # Create the animation
    ani = FuncAnimation(fig, update, frames=p_all.shape[0], blit=True, interval=500, repeat_delay = 1000)

    save_file_name = 'Anim'
    file_dir_anim =  save_file_name + '.mp4'
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=-1)
    ani.save(file_dir_anim, writer=writer)

    plt.show()

type='big_lot'

home_path = os.path.abspath(os.getcwd())

## Load the configuration files
config_path = home_path + '/python/parksim/path_planner/hybrid_astar/Config/'
# config_path = home_path + '/Config/'

with open(config_path + 'config_planner.json') as f:
    config_planner = json.load(f)

with open(config_path + 'config_map.json') as f:
    config_map = json.load(f)

goal_thresh = config_planner['goal_thresh_pos']
Car_obj = Car_class(config_planner)
fig, ax = plt.subplots(figsize=(10, 8))
figa, axa = plt.subplots(figsize=(10, 8))
axes = [ax, axa]
x_min, x_max, y_min, y_max, p_w, p_l, l_w, n_r, n_s, n_s1, obstacleX, obstacleY, s, center_spots, occ_spot_indices, roads_x, roads_y = map_lot(type, config_map, Car_obj, axes)
map_limits = (x_min, x_max, y_min, y_max)

n_spots = center_spots.shape[0]

p = np.array(s)

## Get the complete path of dynamic agents
# Vehicle
dynamic_veh_0 = np.array([np.hstack((center_spots[29], np.deg2rad(0.0))),
                          np.hstack((center_spots[32] + np.array([-Car_obj.length/2-l_w/4, -l_w/2]), np.deg2rad(90.0)))
                          ])
# dynamic_veh_0 = np.array([np.hstack((center_spots[32] + np.array([-Car_obj.length/2-l_w/4, -l_w/2]), np.deg2rad(90.0)))
#                           ])

# center_spots[39] + np.array([-Car_obj.length/2 - l_w/4, p_w]), np.deg2rad(90.0)
dynamic_veh_goal = np.array([np.hstack((center_spots[39] + np.array([-Car_obj.length/2 - l_w/4, p_w]), np.deg2rad(90.0))),
                             np.hstack((center_spots[35], np.deg2rad(0.0)))
                          ])
# dynamic_veh_goal = np.array([np.hstack((center_spots[35], np.deg2rad(0.0)))
#                           ])

# dynamic_veh_0 = np.array([np.hstack((center_spots[35] + np.array([-Car_obj.length/2 - l_w/3, p_w]), np.deg2rad(110.0))),
#                           np.hstack((center_spots[32] + np.array([-Car_obj.length/2-l_w/4, -l_w/2]), np.deg2rad(90.0)))
#                           ])
#
# dynamic_veh_goal = np.array([np.hstack((center_spots[29], np.deg2rad(-180.0))),
#                              np.hstack((center_spots[35], np.deg2rad(0.0)))
#                           ])

dynamic_veh_parking = [0, 1] # 1 is parking/getting out using Hybrid A star, 0 is constant velocity/stationary
T = 100 # total number of time steps to execute

obstacleX_t = copy.deepcopy(obstacleX)
obstacleY_t = copy.deepcopy(obstacleY)

length_preds = 3*T+1 # availability of dynamic vehicle's predictions
dynamic_veh_path = []
ego_obst_x, ego_obst_y = get_vertices_car(Car_obj, p)
obstacleX_dyn = obstacleX + ego_obst_x
obstacleY_dyn = obstacleY + ego_obst_y
for veh_i, veh_parking in enumerate(dynamic_veh_parking):
    if veh_parking:
        path_veh = hybrid_a_star_planning(dynamic_veh_0[veh_i], dynamic_veh_goal[veh_i], obstacleX_dyn, obstacleY_dyn, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)[0]
        if length_preds > len(path_veh.x_list):
            extra_time_steps = int(length_preds - len(path_veh.x_list))
            path_veh_n = np.array([path_veh.x_list, path_veh.y_list, path_veh.yaw_list]).T
            last_state = path_veh_n[-1]
            repeat_veh = np.repeat(last_state.reshape((1, -1)), repeats=extra_time_steps, axis=0)
            path_veh_n = np.vstack((path_veh_n, repeat_veh))
        else:
            path_veh_n = np.array([path_veh.x_list, path_veh.y_list, path_veh.yaw_list]).T
            path_veh_n = path_veh_n[:length_preds]
    else:
        ## dynamic_veh constant velocity
        veh_0 = dynamic_veh_0[veh_i]
        path_veh_n = np.array([veh_0 + i * 0 for i in range(length_preds)])

    dynamic_veh_path.append(path_veh_n)

dynamic_veh_path=np.array(dynamic_veh_path)

# Pedestrian
# ped_0 = np.array([center_spots[28] + np.array([Car_obj.length/2-1, -Car_obj.width/2-1])]) # ped_out
ped_0 = np.array([center_spots[28] + np.array([Car_obj.length/2-2, -Car_obj.width/2-1]),
                  center_spots[27] + np.array([Car_obj.length/2-2, -Car_obj.width/2-1]),
                  center_spots[32] + np.array([-1, -1])]) # ped_in (0 velocity)
# ped_0 = np.array([np.array([-Car_obj.length/2-2, -Car_obj.width/2-1])]) # no_ped
ped_vel = np.array([[-0.0, -0.0],
                    [-0.0, -0.0],
                    [0.0, 0.0]
                    ])
# time steps
delay = 5 # should be less than or equal to T, length_preds = T+1
moving_time = int(length_preds - delay)

ped_init = np.array([np.repeat(np.array([ped_0[j]]), repeats = delay, axis=0) for j in range(ped_0.shape[0])])
ped_path = np.array([np.vstack((ped_init[j], np.array([ped_0[j] + MOTION_RESOLUTION*i*ped_vel[j] for i in range(moving_time)]))) for j in range(len(ped_0))])

## construct KDTree for static obstacles
static_xy = np.array([obstacleX, obstacleY]).T
static_obs_kd_tree = cKDTree(static_xy)

Sigma_0 = np.repeat(np.array([[[0.5 * Car_obj.length, 0], [0, 0.5 * Car_obj.width]]]), repeats=dynamic_veh_0.shape[0],
                    axis=0)  # Covariance for each vehicle
Sigma_0_ped = np.repeat(np.array([[[0.01 * PED_RAD, 0], [0, 0.01 * PED_RAD]]]), repeats=ped_0.shape[0],
                        axis=0)  # Covariance for each pedestrian
Q = np.array([[0.5, 0], [0, 0.5]])  # Process noise (uncertainty growth)

wb_2 = Car_obj.wheelBase / 2

time_pred = []
time_strat = []
n_sims = 1
for _ in range(n_sims):

    t = 0
    reached_spot = False
    p = np.array(s)
    p_all = [p]

    P_O_all_spots = 0.5*np.ones(n_spots)

    while t < T and (not reached_spot):

        T_pred = 5 # time steps
        dynamic_veh_path_t = dynamic_veh_path[:, t:t+T_pred+1]
        ped_path_t = ped_path[:, t:t+T_pred+1]

        obstacleX_t = copy.deepcopy(obstacleX)
        obstacleY_t = copy.deepcopy(obstacleY)

        # for veh_i in range(dynamic_veh_path_t.shape[0]):
        #     dyn_X, dyn_Y = get_points_car(Car_obj, dynamic_veh_path_t[veh_i, 0])
        #     obstacleX_t = obstacleX_t + dyn_X
        #     obstacleY_t = obstacleY_t + dyn_Y

        # for ped_i in range(ped_path_t.shape[0]):
        #     dyn_X, dyn_Y = get_ped_points(PED_RAD, ped_path_t[ped_i, 0])
        #     obstacleX_t = obstacleX_t + dyn_X
        #     obstacleY_t = obstacleY_t + dyn_Y

        # plot_ell(s, Car_obj, ax)
        
        observed_spots, explore_points, observed_p_spots, dist_inf_norm, p_c = rect_dist_obs_p_spots_plot(p, center_spots, roads_x, roads_y, map_limits, Car_obj, ax)
        # print("Observed Spots: ", observed_spots)

        unobserved_spots = list(set(np.arange(n_spots).tolist()) - (set(observed_spots.tolist()) | set(observed_p_spots.tolist())))

        P_O_unobserved_spots = update_spots_occupancy_arr_dep(P_O_all_spots[unobserved_spots], T_pred) # T x n_unobserved_spots

        P_O_all_spots[unobserved_spots] = P_O_unobserved_spots[-1]

        occ_spots, vac_spots, occ_spots_dyn, occ_spots_veh, occ_spots_ped = get_occ_vac_spots(static_obs_kd_tree, dynamic_veh_path_t[:, 0, :], ped_path_t[:, 0, :], center_spots, observed_spots, Car_obj, p_l, p_w)
        occ_p_spots, vac_p_spots, occ_p_spots_dyn, occ_p_spots_veh, occ_p_spots_ped = get_occ_vac_spots(static_obs_kd_tree, dynamic_veh_path_t[:, 0, :], ped_path_t[:, 0, :], center_spots, observed_p_spots, Car_obj, p_l, p_w)
        
        # print(f"Occupied spots: {occ_spots}, Vacant Spots: {vac_spots}")
        # print(f"Occupied by Vehicle: {occ_spots_veh}, Occupied by Pedestrian: {occ_spots_ped}")
        start_prob = time.time()

        if len(occ_spots) > 0:
            P_O_all_spots[occ_spots] = 1
            P_O_occ_spots = update_spots_occupancy_arr_dep(P_O_all_spots[occ_spots], T_pred)

            P_O_all_spots[occ_spots] = P_O_occ_spots[-1]

        if len(occ_p_spots) > 0:
            P_O_all_spots[occ_p_spots] = 1
            P_O_occ_p_spots = update_spots_occupancy_arr_dep_p_o(P_O_all_spots[occ_p_spots], p_c[occ_p_spots], T_pred)

            P_O_all_spots[occ_p_spots] = P_O_occ_p_spots[-1]             

        results = []
        test_spots_ind = np.array([])    

        if (len(vac_p_spots) + len(occ_p_spots_dyn) > 0) or (len(vac_spots) + len(occ_spots_dyn) > 0):
            if (len(vac_p_spots) + len(occ_p_spots_dyn) > 0):
                P_O_p_vacant, P_O_p_occ = occupancy_probability_multiple_spots_occ_dep_p(T_pred, dynamic_veh_path_t, ped_path_t, Sigma_0, Sigma_0_ped, Q, center_spots, vac_p_spots, occ_p_spots_veh, occ_p_spots_ped, p_c, Car_obj)

                P_O_all_spots[vac_p_spots] = P_O_p_vacant[-1]
                P_O_all_spots[occ_p_spots_dyn] = P_O_p_occ[-1]

                ## Choose which spots to test for HA* paths
                prob_thresh_vac = 0.3
                vacant_spots_p_vacant_ind = np.where(P_O_p_vacant[-1] <= prob_thresh_vac)[0]
                vacant_spots_p_vacant = np.array(vac_p_spots)[vacant_spots_p_vacant_ind]
                prob_thresh_occ = 0.7
                vacant_spots_p_occ_ind = np.where(P_O_p_occ[-1] <= prob_thresh_occ)[0]
                vacant_spots_p_occ = np.array(occ_p_spots_dyn)[vacant_spots_p_occ_ind]

                test_spots_ind = np.hstack((test_spots_ind, vacant_spots_p_vacant, vacant_spots_p_occ))

            if (len(vac_spots) + len(occ_spots_dyn) > 0):
                P_O_vacant, P_O_occ = occupancy_probability_multiple_spots_occ_dep(T_pred, dynamic_veh_path_t, ped_path_t, Sigma_0, Sigma_0_ped, Q, center_spots, vac_spots, occ_spots_veh, occ_spots_ped, Car_obj)

                P_O_all_spots[vac_spots] = P_O_vacant[-1]
                P_O_all_spots[occ_spots_dyn] = P_O_occ[-1]

                time_pred.append(time.time() - start_prob)

                ## Choose which spots to test for HA* paths
                prob_thresh_vac = 0.3
                vacant_spots_vacant_ind = np.where(P_O_vacant[-1] <= prob_thresh_vac)[0]
                vacant_spots_vacant = np.array(vac_spots)[vacant_spots_vacant_ind]
                prob_thresh_occ = 0.7
                vacant_spots_occ_ind = np.where(P_O_occ[-1] <= prob_thresh_occ)[0]
                vacant_spots_occ = np.array(occ_spots_dyn)[vacant_spots_occ_ind]

                test_spots_ind = np.hstack((test_spots_ind, vacant_spots_vacant, vacant_spots_occ))

            test_spots_center = center_spots[test_spots_ind.astype(int)]

            goal_park_spots = []  # park_spot i ->  goal yaw = 0.0 if 0, and np.pi if 1 -> (x, y, yaw) of goal
            for spot_xy in test_spots_center.tolist():
                # transforming center of parking spot to rear axle of vehicle (goal x, y) with appropriate goal yaw
                goal1 = np.array([spot_xy[0] - Car_obj.length / 2 + Car_obj.axleToBack, spot_xy[1], 0.0])
                goal2 = np.array([spot_xy[0] + Car_obj.length / 2 - Car_obj.axleToBack, spot_xy[1], np.pi])
                goal_spot_xy = [goal1, goal2]
                goal_plot = goal1
                goal_park_spots.append(goal_spot_xy)

            goal_park_spots = list(chain.from_iterable(goal_park_spots))
            # transforming to center of vehicle
            g_list = [[g[0] + wb_2 * np.cos(g[2]), g[1] + wb_2 * np.sin(g[2]), g[2]] for g in goal_park_spots]
            
            results_raw = parallel_run(p, obstacleX_t, obstacleY_t, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list)

            results = [result[0] for result in results_raw if result[-1]]

        else:            
            time_pred.append(0.0)
        
        # print(f"State of ego: {p}, at time: {t}")

        # if t>=4:
        #     print("Stop!")

        entropy_before = (-P_O_all_spots*np.log(P_O_all_spots)-(1-P_O_all_spots)*np.log(1-P_O_all_spots))/np.log(2)
        P_O_all_spots_after = copy.deepcopy(P_O_all_spots)

        T_explore = 2
        start_strat = time.time()
        if results:
            path_list = [np.array([path.x_list, path.y_list, path.yaw_list]).T for path in results]
            longest_path_to_spot = max([len(path) for path in path_list])
            length_path_to_eval = int(max(T_pred, longest_path_to_spot) + MAX_WAIT_TIME / MOTION_RESOLUTION)

            dynamic_veh_path_t_eval = dynamic_veh_path[:, t:t+length_path_to_eval]
            ped_path_t_eval = ped_path[:, t:t+length_path_to_eval]

            start_t_c = time.time()
            costs = []
            collisions = []
            wait_times = []
            path_eval = []
            for path in path_list:
                path_current, cost_current, collision_current, wait_time_current = evaluate_path(path, obstacleX_t, obstacleY_t, dynamic_veh_path_t_eval, ped_path_t_eval)
                if collision_current:
                    costs.append(np.inf)
                else:
                    costs.append(cost_current)
                collisions.append(collision_current)
                wait_times.append(wait_time_current)
                path_eval.append(path_current)

            # print("Comp time cost: ", time.time() - start_t_c)

            if all(collisions):
                # explore_y = np.min(center_spots[observed_spots, 1]) + 4.0
                # explore_point = [p[0], explore_y, p[2]]
                # results_raw = [hybrid_a_star_planning(p, explore_point, obstacleX, obstacleY, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)]
                # results = [result[0] for result in results_raw if result[-1]]
                # straight_goal = dynamic_veh_0[0] + np.array([0.0, -15.0, 0.0])
                # path_list_ex = [np.array([p + (i*MOTION_RESOLUTION) * np.array([0.0, -2.0, 0.0]) for i in range(int(T_explore/MOTION_RESOLUTION))])]
                
                g_list_ex = [[explore_points[i, 0], explore_points[i, 1], explore_points[i, 2]] for i in range(explore_points.shape[0])]

                path_list_ex_s = [np.array(spline_inter(p, g_list_ex[0])).T] # go straight spline
                # path_list_ex = [np.array(spline_inter(p, g_ex)).T for g_ex in g_list_ex] # spline for all
                # path_list_ex_s = [np.array([p + (i*MOTION_RESOLUTION) * np.array([0.0, -2.0, 0.0]) for i in range(int(T_explore/MOTION_RESOLUTION))])] # CV                 
                results_raw_ex = parallel_run(p, obstacleX_t, obstacleY_t, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list_ex[1:]) # ignoring straight goal
                results_ex = [result[0] for result in results_raw_ex if result[-1]]
                path_list_ex = path_list_ex_s + [np.array([path.x_list, path.y_list, path.yaw_list]).T for path in results_ex]
                if path_list_ex:
                    # path_list_ex = [[np.array([path.x_list, path.y_list, path.yaw_list]).T] for path in results]
                    longest_path_to_spot = max([len(path) for path in path_list_ex])
                    length_path_to_eval = int(max(T_pred, longest_path_to_spot) + MAX_WAIT_TIME / MOTION_RESOLUTION)

                    dynamic_veh_path_t_eval = dynamic_veh_path[:, t:t + length_path_to_eval]
                    ped_path_t_eval = ped_path[:, t:t + length_path_to_eval]
                    costs_ex = []
                    collisions_ex = []
                    wait_times_ex = []
                    path_eval_ex = []
                    for path in path_list_ex:
                        path_current, cost_current, collision_current, wait_time_current = evaluate_path(path, obstacleX_t, obstacleY_t,
                                                                                            dynamic_veh_path_t_eval, ped_path_t_eval)
                        costs_ex.append(cost_current)
                        collisions_ex.append(collision_current)
                        wait_times_ex.append(wait_time_current)
                        path_eval_ex.append(path_current)

                    if not all(collisions_ex):
                        print("Exploring inspite of available spots")
                        no_coll_path_ind = [i for i, x in enumerate(collisions_ex) if not x] # indices of paths that are not colliding
                        # no_coll_path_goals = [path_eval_ex[i][-1] for i in no_coll_path_ind] # paths that are not colliding
                        
                        for g_e_i in no_coll_path_ind:
                            g_e = path_eval_ex[g_e_i][-1]
                            observed_spots_e, explore_points_e, observed_p_spots_e, dist_inf_norm_e, p_c_e = rect_dist_obs_p_spots_e(g_e, center_spots, roads_x, roads_y, map_limits, Car_obj)
                            # print("Observed Spots: ", observed_spots)

                            unobserved_spots_e = list(set(np.arange(n_spots).tolist()) - (set(observed_spots_e.tolist()) | set(observed_p_spots_e.tolist())))

                            P_O_unobserved_spots_after = update_spots_occupancy_arr_dep(P_O_all_spots_after[unobserved_spots_e], T) # T x n_unobserved_spots
                            P_O_all_spots_after[unobserved_spots_e] = P_O_unobserved_spots_after[-1]
    
                            occ_p_spots_e, vac_p_spots_e = get_occ_vac_spots_stat(static_obs_kd_tree, center_spots, observed_p_spots_e, Car_obj, p_l, p_w)

                            P_O_p_observed_o_spots_after = update_spots_occupancy_arr_dep_p_o(P_O_all_spots_after[occ_p_spots_e], p_c_e[occ_p_spots_e], T)
                            P_O_all_spots_after[occ_p_spots_e] = P_O_p_observed_o_spots_after[-1]

                            P_O_p_observed_v_spots_after = update_spots_occupancy_arr_dep_p_v(P_O_all_spots_after[vac_p_spots_e], p_c_e[vac_p_spots_e], T)
                            P_O_all_spots_after[vac_p_spots_e] = P_O_p_observed_v_spots_after[-1]

                            entropy_after = (-P_O_all_spots_after*np.log(P_O_all_spots_after)-(1-P_O_all_spots_after)*np.log(1-P_O_all_spots_after))/np.log(2)
                            entropy_after[observed_spots_e] = 0.0

                            fract_IG = 1*(np.sum(entropy_before) - np.sum(entropy_after)) # /np.sum(entropy_before)
                            costs_ex[g_e_i] -= fract_IG

                        best_path_ex = path_eval_ex[np.argmin(costs_ex)]
                        p_all = np.vstack((p_all, best_path_ex))
                        t += best_path_ex.shape[0]
                    else:
                        print("Stationary")
                        p_all = np.vstack((p_all, p[None, :]))
                        t += 1
                else:
                    print("Stationary")
                    p_all = np.vstack((p_all, p[None, :]))
                    t += 1

            else:
                print("Go to spot")
                best_path = path_eval[np.argmin(costs)]
                p_all = np.vstack((p_all, best_path))
                t += best_path.shape[0]

        else:
            # explore_y = np.min(center_spots[observed_spots, 1]) + 4.0
            # explore_point = [p[0], explore_y, p[2]]
            # results_raw = [hybrid_a_star_planning(p, explore_point, obstacleX, obstacleY, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)]
            # results = [result[0] for result in results_raw if result[-1]]
            # path_list_ex = [np.array([p + (i*MOTION_RESOLUTION) * np.array([0.0, -2.0, 0.0]) for i in range(int(T_explore/MOTION_RESOLUTION))])]
            
            g_list_ex = [[explore_points[i, 0], explore_points[i, 1], explore_points[i, 2]] for i in range(explore_points.shape[0])]
            path_list_ex_s = [np.array(spline_inter(p, g_list_ex[0])).T] # go straight spline
            # path_list_ex = [np.array(spline_inter(p, g_ex)).T for g_ex in g_list_ex] # spline for all
            # path_list_ex_s = [np.array([p + (i*MOTION_RESOLUTION) * np.array([0.0, -2.0, 0.0]) for i in range(int(T_explore/MOTION_RESOLUTION))])] # CV               
            results_raw_ex = parallel_run(p, obstacleX_t, obstacleY_t, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, g_list_ex[1:]) # ignoring straight goal
            results_ex = [result[0] for result in results_raw_ex if result[-1]]
            path_list_ex = path_list_ex_s + [np.array([path.x_list, path.y_list, path.yaw_list]).T for path in results_ex]
            if path_list_ex:
                # path_list_ex = [[np.array([path.x_list, path.y_list, path.yaw_list]).T] for path in results]
                longest_path_to_spot = max([len(path) for path in path_list_ex])
                length_path_to_eval = int(max(T_pred, longest_path_to_spot) + MAX_WAIT_TIME / MOTION_RESOLUTION)

                dynamic_veh_path_t_eval = dynamic_veh_path[:, t:t + length_path_to_eval]
                ped_path_t_eval = ped_path[:, t:t + length_path_to_eval]
                costs_ex = []
                collisions_ex = []
                wait_times_ex = []
                path_eval_ex = []
                for path in path_list_ex:
                    path_current, cost_current, collision_current, wait_time_current = evaluate_path(path, obstacleX_t, obstacleY_t,
                                                                                        dynamic_veh_path_t_eval,
                                                                                        ped_path_t_eval)
                    costs_ex.append(cost_current)
                    collisions_ex.append(collision_current)
                    wait_times_ex.append(wait_time_current)
                    path_eval_ex.append(path_current)
                
                if not all(collisions_ex):
                    print("Exploring coz no spots")

                    no_coll_path_ind = [i for i, x in enumerate(collisions_ex) if not x] # indices of paths that are not colliding
                    # no_coll_path_goals = [path_eval_ex[i][-1] for i in no_coll_path_ind] # paths that are not colliding
                    
                    for g_e_i in no_coll_path_ind:
                        g_e = path_eval_ex[g_e_i][-1]
                        observed_spots_e, explore_points_e, observed_p_spots_e, dist_inf_norm_e, p_c_e = rect_dist_obs_p_spots_e(g_e, center_spots, roads_x, roads_y, map_limits, Car_obj)
                        # print("Observed Spots: ", observed_spots)

                        unobserved_spots_e = list(set(np.arange(n_spots).tolist()) - (set(observed_spots_e.tolist()) | set(observed_p_spots_e.tolist())))

                        P_O_unobserved_spots_after = update_spots_occupancy_arr_dep(P_O_all_spots_after[unobserved_spots_e], T) # T x n_unobserved_spots
                        P_O_all_spots_after[unobserved_spots_e] = P_O_unobserved_spots_after[-1]

                        occ_p_spots_e, vac_p_spots_e = get_occ_vac_spots_stat(static_obs_kd_tree, center_spots, observed_p_spots_e, Car_obj, p_l, p_w)

                        P_O_p_observed_o_spots_after = update_spots_occupancy_arr_dep_p_o(P_O_all_spots_after[occ_p_spots_e], p_c_e[occ_p_spots_e], T)
                        P_O_all_spots_after[occ_p_spots_e] = P_O_p_observed_o_spots_after[-1]

                        P_O_p_observed_v_spots_after = update_spots_occupancy_arr_dep_p_v(P_O_all_spots_after[vac_p_spots_e], p_c_e[vac_p_spots_e], T)
                        P_O_all_spots_after[vac_p_spots_e] = P_O_p_observed_v_spots_after[-1]
                    
                        entropy_after = (-P_O_all_spots_after*np.log(P_O_all_spots_after)-(1-P_O_all_spots_after)*np.log(1-P_O_all_spots_after))/np.log(2)
                        entropy_after[observed_spots_e] = 0.0

                        fract_IG = 1*(np.sum(entropy_before) - np.sum(entropy_after)) # /np.sum(entropy_before)
                        costs_ex[g_e_i] -= fract_IG

                    best_path_ex = path_eval_ex[np.argmin(costs_ex)]
                    p_all = np.vstack((p_all, best_path_ex))
                    t += best_path_ex.shape[0]
                else:
                    print("Stationary")
                    p_all = np.vstack((p_all, p[None, :]))
                    t += 1
            else:
                print("Stationary")
                p_all = np.vstack((p_all, p[None, :]))
                t += 1

        time_strat.append(time.time() - start_strat)
        p = p_all[-1]
        if len(test_spots_ind):
            reached_spot = np.min(np.linalg.norm(test_spots_center - p[None, :2], axis=1)) < goal_thresh

print(f"Belief Predictions Computation time: Mean = {np.mean(time_pred)}, STD={np.std(time_pred)}")
print(f"Strategy Planner Computation time: Mean = {np.mean(time_strat)}, STD={np.std(time_strat)}")
# Plotting the dynamic agents
for time_dynamic in range(t+1):
    # plot vehicle
    for veh_i in range(dynamic_veh_path.shape[0]):
        p_yaw = dynamic_veh_path[veh_i, time_dynamic, 2]
        p_x = dynamic_veh_path[veh_i, time_dynamic, 0]
        p_y = dynamic_veh_path[veh_i, time_dynamic, 1]
        plot_other_car_trans(p_x, p_y, p_yaw, ax)

    # plot pedestrians
    for ped_i in range(ped_path.shape[0]):
        circle = Circle((ped_path[ped_i, time_dynamic, 0], ped_path[ped_i, time_dynamic, 1]), radius=PED_RAD,
                        facecolor='red', alpha=0.1)
        ax.add_artist(circle)

    plot_car_trans(p_all[time_dynamic, 0], p_all[time_dynamic, 1], p_all[time_dynamic, 2], ax)

observed_spots, explore_points = rect_dist_obs_spots_plot(p, center_spots, roads_x, roads_y, map_limits, Car_obj, ax)
for ax in axes:
    ax.axis('equal')

plot_anim(axa, figa, p_all, dynamic_veh_path, ped_path)

plt.show()