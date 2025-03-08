import numpy as np
import os
import json
from itertools import chain
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time

from parksim.path_planner.hybrid_astar.hybrid_a_star_parallel import map_lot, Car_class, hybrid_a_star_planning, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, MOTION_RESOLUTION
from parksim.path_planner.hybrid_astar.car import plot_car, plot_other_car

def mahalanobis_distance(mu, Sigma, park_spot, Car_obj):
    """
    Computes the Mahalanobis distance from the mean position `mu` to the nearest
    boundary of the rectangular parking spot.

    Parameters:
    - mu: position of vehicle (2, )
    - Sigma: covariance matrix of the vehicles' positions (2 x 2 array)
    - park_spot: center(x, y) of parking spot and whether parking spot is left (1) or right (0) to center lane  (3, )
    - Car_obj: object of Car_class that stores the dimensions of car

    Returns:
    - mahalanobis_distance/probability
    """

    x_min = park_spot[0]-Car_obj.length/2
    x_max = park_spot[0] + Car_obj.length / 2
    y_min = park_spot[1] - Car_obj.width / 2
    y_max = park_spot[1] + Car_obj.width / 2

    # Compute Mahalanobis distance to each boundary
    d_x_min = (x_min - mu[0]) / np.sqrt(Sigma[0, 0])
    d_x_max = (x_max - mu[0]) / np.sqrt(Sigma[0, 0])
    d_y_min = (y_min - mu[1]) / np.sqrt(Sigma[1, 1])
    d_y_max = (y_max - mu[1]) / np.sqrt(Sigma[1, 1])

    # Find minimum Mahalanobis distance to any boundary
    # d_min = min(abs(d_x_min), abs(d_x_max), abs(d_y_min), abs(d_y_max))

    d_min = distance.mahalanobis(park_spot[:2], mu[:2], Sigma)

    prob = np.exp(-0.1*d_min)
    # prob = (norm.cdf(d_x_max) - norm.cdf(d_x_min))*(norm.cdf(d_y_max) - norm.cdf(d_y_min))

    return prob

## Investigate errors after computing Sigma_t
def occ_vec(T, dynamic_veh_path, Sigma_0, Q, park_spots_xy, Car_obj):
    n_spots = park_spots_xy.shape[0]
    n_vehicles = dynamic_veh_path.shape[0]
    P_O = np.zeros((T + 1, n_spots))
    P_O_d = np.zeros((T + 1, n_spots))
    P_O_result = np.zeros((T + 1, n_spots))

    yaw_t = dynamic_veh_path[:, :, 2]
    cos_theta = np.cos(yaw_t)
    sin_theta = np.sin(yaw_t)
    rotation_matrices = np.stack([
        np.stack([cos_theta, -sin_theta], axis=-1),
        np.stack([sin_theta, cos_theta], axis=-1)
    ], axis=-2)

    Sigma_t = np.einsum('ntij,njk,ntkl->ntil', rotation_matrices, Sigma_0 + Q, rotation_matrices)
    prob_0 = np.array([[mahalanobis_distance(dynamic_veh_path[i, 0], Sigma_t[i], park_spots_xy[j], Car_obj)
                        for i in range(n_vehicles)] for j in range(n_spots)])
    P_O[0] = 1 - np.prod(1 - prob_0, axis=1)
    P_O_d[0] = P_O[0]

    mu_t = dynamic_veh_path[:, :, :2]
    dynamic_veh_vel = np.diff(mu_t, axis=1) / MOTION_RESOLUTION
    vel_t = np.pad(dynamic_veh_vel, ((0, 0), (1, 0), (0, 0)), mode='edge')

    vectors_to_spots = park_spots_xy[:, np.newaxis, np.newaxis, :2] - mu_t[np.newaxis, :, :, :2]
    vel_norm = np.linalg.norm(vel_t, axis=2, keepdims=True)
    vectors_norm = np.linalg.norm(vectors_to_spots, axis=3, keepdims=True)
    epsilon = 1e-6
    vel_norm = np.maximum(vel_norm, epsilon)
    vectors_norm = np.maximum(vectors_norm, epsilon)

    vel_normalized = vel_t / vel_norm
    vectors_normalized = vectors_to_spots / vectors_norm
    cosine_similarity = np.einsum('tij,mtij->mti', vel_normalized, vectors_normalized)
    cosine_distance = 1 - cosine_similarity
    zero_spot_veh = (vel_norm[..., 0] <= epsilon) | (vectors_norm[..., 0] <= epsilon)
    cosine_distance[zero_spot_veh] = 0.0

    rho = 0.8
    Sigma_vel_t = (2 * Sigma_t + Q - 2 * np.array([[rho, 0.0], [0.0, rho]])) / MOTION_RESOLUTION ** 2
    veh_vel_uncertainty = np.trace(Sigma_vel_t, axis1=2, axis2=3)

    prob_t_dist = np.array([[mahalanobis_distance(mu_t[i, t], Sigma_t[i], park_spots_xy[j], Car_obj)
                             for i in range(n_vehicles)] for j in range(n_spots) for t in range(T + 1)]).reshape(T + 1,
                                                                                                                 n_spots,
                                                                                                                 n_vehicles)
    prob_t = prob_t_dist * np.exp(-0.1 * cosine_distance / veh_vel_uncertainty[np.newaxis, :, :])
    prob_t_new = 1 - np.prod(1 - prob_t, axis=2)

    alpha = 0.95
    P_O[1:] = alpha * P_O[:-1] + prob_t_new[1:] * (1 - P_O[:-1])
    P_O_d[1:] = 1 - (alpha * (1 - P_O_d[:-1]) + (1 - prob_t_new[1:]) * P_O_d[:-1])
    P_O_result[:, :2] = P_O_d[:, :2]

    return P_O_result

def occupancy_probability_multiple_spots(T, dynamic_veh_path, Sigma_0, Q, park_spots_xy, Car_obj):
    """
    Computes the recursive occupancy probability P(O_t) for multiple vehicles and multiple parking spots.

    Parameters:
    - T: Total number of time steps
    - dynamic_veh_path: path of dynamic vehicle's mean (n_vehicles x length of path x 3 array)
    - Sigma_0: Initial covariance matrix of the vehicles' positions (n_vehicles x 2 x 2 array)
    - Q: Process noise covariance matrix (2 x 2)
    - park_spots_xy: Array of centers(x, y) of parking spots and whether parking spot is left (1) or right (0) to center lane  (n_spots x 3)
    - Car_obj: object of Car_class that stores the dimensions of car

    Returns:
    - P_O: Occupancy probability over time for each parking spot.
    """
    n_spots = park_spots_xy.shape[0]
    n_vehicles = dynamic_veh_path.shape[0]
    P_O = np.zeros((T + 1, n_spots))  # Occupancy probability for each time step and parking spot
    P_O_d = np.zeros((T + 1, n_spots))  # Occupancy probability for each time step and parking spot, focussed on departing vehicles
    P_O_result = np.zeros((T + 1, n_spots))  # Occupancy probability for each time step and parking spot, focussed on departing vehicles

    # Initialize with multiple vehicles' initial probability of occupying the spots
    yaw_t = dynamic_veh_path[:, 0, 2]
    cos_theta = np.cos(yaw_t)
    sin_theta = np.sin(yaw_t)
    rotation_matrices = np.stack([
        np.stack([cos_theta, -sin_theta], axis=-1),
        np.stack([sin_theta, cos_theta], axis=-1)
    ], axis=-2)
    Sigma_t = np.array([rotation_matrices[i] @ (Sigma_0[i] + Q) @ rotation_matrices[i].T  for i in range(Sigma_0.shape[0])])  # Update covariances
    prob_0 = np.array([[mahalanobis_distance(dynamic_veh_path[i, 0], Sigma_t[i], park_spots_xy[j], Car_obj) for i in range(n_vehicles)] for j in range(n_spots)])
    P_O[0] = 1 - np.prod(1 - prob_0, axis=1)  # Probability that at least one vehicle occupies each spot
    P_O_d[0] = 1 - np.prod(1 - prob_0, axis=1)
    mu_t = dynamic_veh_path[:, 0]
    Sigma_t = Sigma_0

    dynamic_veh_vel = np.diff(dynamic_veh_path[:, :, :2], axis=1) / MOTION_RESOLUTION

    # Iterate through time steps t = 1 to T
    for t in range(1, T + 1):
        # Propagate mean and covariance for each vehicle
        mu_t = dynamic_veh_path[:, t, :2]  # Update positions (vectorized)
        yaw_t = dynamic_veh_path[:, t, 2]

        vel_t = dynamic_veh_vel[:, min(t, T-1), :2] # n_vehicles x 2
        vectors_to_spots = park_spots_xy[:, np.newaxis, :2] - mu_t[np.newaxis, :, :2] # n_spots x n_vehicles x 2
        # Compute norms with thresholding
        vel_norm = np.linalg.norm(vel_t, axis=1, keepdims=True)
        vectors_norm = np.linalg.norm(vectors_to_spots, axis=2, keepdims=True)
        epsilon = 1e-6
        vel_norm = np.maximum(vel_norm, epsilon)
        vectors_norm = np.maximum(vectors_norm, epsilon)
        # Normalize the vectors
        vel_normalized = vel_t / vel_norm
        vectors_normalized = vectors_to_spots / vectors_norm
        # Compute cosine similarity
        dot_prod = np.einsum('ij,mij->mi', vel_t, vectors_to_spots)
        cosine_similarity = np.einsum('ij,mij->mi', vel_normalized, vectors_normalized)
        zero_spot_veh = np.logical_or(vel_norm[:, 0] <= epsilon, vectors_norm[:, :, 0] <= epsilon)
        spot_ind, veh_ind = np.where(zero_spot_veh)
        # Convert to cosine distance (1 - cosine similarity)
        cosine_distance = 1 - cosine_similarity # (n_spots, n_vehicles)
        cosine_distance[spot_ind, veh_ind] = 0.0
        dot_prod[spot_ind, veh_ind] = 100000.0

        cos_theta = np.cos(yaw_t)
        sin_theta = np.sin(yaw_t)
        rotation_matrices = np.stack([
            np.stack([cos_theta, -sin_theta], axis=-1),
            np.stack([sin_theta, cos_theta], axis=-1)
        ], axis=-2)
        Sigma_t = np.array([rotation_matrices[i] @ (Sigma_t[i] + Q) @ rotation_matrices[i].T  for i in range(Sigma_t.shape[0])])  # Update covariances

        rho = 0.0

        Sigma_vel_t = (2*Sigma_t + Q - 2*np.array([[rho, 0.0],
                                                   [0.0, rho]]))/MOTION_RESOLUTION**2
        veh_vel_uncertainty = np.trace(Sigma_vel_t, axis1=1, axis2=2) # (n_vehicles,)
        # Compute probability of being inside parking spot for each vehicle and each spot (n_spots x n_vehicles)
        prob_t_dist = np.array([[mahalanobis_distance(mu_t[i], Sigma_t[i], park_spots_xy[j], Car_obj) for i in range(n_vehicles)] for j in range(n_spots)])
        prob_t =prob_t_dist*(1/(1+np.exp(-0.001*dot_prod/veh_vel_uncertainty[np.newaxis, :])))# including velocity information
        # prob_t = prob_t_dist

        prob_t_new = 1 - np.prod(1 - prob_t, axis=1)  # Probability that at least one vehicle occupies each spot using only dynamic obstacles
        alpha = 0.95 # filter: exponential distribution for departure rate, poisson for arrival rate
        P_O[t] = alpha * P_O[t - 1] + prob_t_new * (1 - P_O[t - 1])  # At least one vehicle per spot
        # P_O[t] = prob_t_new

        P_O_d[t] = 1-(alpha * (1-P_O_d[t - 1]) + (1-prob_t_new) * (P_O_d[t - 1]))

        # P_O[t] = prob_t1
        # P_enter_t = norm.cdf(d_t)  # Probability vehicle i enters spot j
        #
        # # Update occupancy probability recursively for all vehicles and spots
        # P_O[t] = alpha * P_O[t - 1] + (1 - alpha) * (
        #             1 - np.prod(1 - P_enter_t, axis=0))  # At least one vehicle per spot

    # P_O_result[:, 0] =  P_O[:, 0]
    # P_O_result[:, 1] = P_O_d[:, 1]

    P_O_result[:, 0] =  P_O[:, 0]
    P_O_result[:, 1] = P_O[:, 1]
    return P_O_result

def occupancy_probability_multiple_spots_occ_dep(T, dynamic_veh_path_t, ped_path_t, Sigma_0, Sigma_0_ped, Q, center_spots, vac_spots, occ_spots_veh, occ_spots_ped, Car_obj):
    """
    Computes the recursive occupancy probability P(O_t) for multiple vehicles and multiple parking spots.

    Parameters:
    - T: Total number of time steps
    - dynamic_veh_path_t: path of dynamic vehicle's mean (n_vehicles x length of path x 3 array)
    - ped_path_t: path of pedestrian's mean (n_ped x length of path x 2 array)
    - Sigma_0, Sigma_0_ped: Initial covariance matrix of the vehicles' and pedestrians' positions (n x 2 x 2 array)
    - Q: Process noise covariance matrix (2 x 2)
    - center_spots: Array of centers(x, y) of all parking spots
    - vac_spots, occ_spots_veh, occ_spots_ped: indices of vacant, occupied spots by dynamic vehicles and pedestrians
    - Car_obj: object of Car_class that stores the dimensions of car

    Returns:
    - P_0_vacant, P_0_occ_veh, P_0_occ_ped: Occupancy probability over time for each parking spot.
    """
    n_vehicles = dynamic_veh_path.shape[0]
    P_0_vacant = np.zeros((T + 1, len(vac_spots)))  # Occupancy probability for each time step and parking spot
    P_0_occ_veh = np.zeros((T + 1, len(occ_spots_veh)))  # Occupancy probability for each time step and parking spot, focussed on departing vehicles
    P_0_occ_ped = np.zeros((T + 1, len(occ_spots_ped)))  # Occupancy probability for each time step and parking spot, focussed on departing pedestrians

    # Initialize with multiple vehicles' initial probability of occupying the spots
    yaw_t = dynamic_veh_path_t[:, 0, 2]
    cos_theta = np.cos(yaw_t)
    sin_theta = np.sin(yaw_t)
    rotation_matrices = np.stack([
        np.stack([cos_theta, -sin_theta], axis=-1),
        np.stack([sin_theta, cos_theta], axis=-1)
    ], axis=-2)
    Sigma_t = np.array([rotation_matrices[i] @ (Sigma_0[i] + Q) @ rotation_matrices[i].T  for i in range(Sigma_0.shape[0])])  # Update covariances for dynamic veh
    Sigma_t_ped = np.array([Sigma_0_ped[i] + Q  for i in range(Sigma_0_ped.shape[0])]) # Update covariances for ped

    Sigma_t_all = np.vstack((Sigma_t, Sigma_t_ped))
    n_agents = Sigma_t_all.shape[0]
    park_spots_xy = np.vstack((center_spots[vac_spots], center_spots[occ_spots_veh], center_spots[occ_spots_ped]))
    n_spots = park_spots_xy.shape[0]

    agent_path = np.vstack((dynamic_veh_path_t[:, :, :2], ped_path_t))

    prob_0 = np.array([[mahalanobis_distance(agent_path[i, 0], Sigma_t_all[i], park_spots_xy[j], Car_obj) for i in range(n_agents)] for j in range(n_spots)]) # (n_spots, n_agents)
    P_0_all = 1 - np.prod(1 - prob_0, axis=1)  # Probability that at least one vehicle occupies each spot
    P_0_vacant = P_0_all[:len(vac_spots)]
    P_0_occ_veh = P_0_all[len(vac_spots):len(vac_spots)+len(occ_spots_veh)]
    P_0_occ_ped = P_0_all[len(vac_spots)+len(occ_spots_veh):len(vac_spots) + len(occ_spots_veh)+len(occ_spots_ped)]

    mu_t = dynamic_veh_path[:, 0]
    Sigma_t = Sigma_0

    dynamic_veh_vel = np.diff(dynamic_veh_path_t[:, :, :2], axis=1) / MOTION_RESOLUTION
    ped_vel = np.diff(ped_path_t[:, :, :2], axis=1) / MOTION_RESOLUTION

    # Iterate through time steps t = 1 to T
    for t in range(1, T + 1):
        # Propagate mean and covariance for each vehicle
        mu_t = dynamic_veh_path_t[:, t, :2]  # Update positions (vectorized)
        yaw_t = dynamic_veh_path[:, t, 2]

        vel_t = dynamic_veh_vel[:, min(t, T-1), :2] # n_vehicles x 2
        vectors_to_spots = park_spots_xy[:, np.newaxis, :2] - mu_t[np.newaxis, :, :2] # n_spots x n_vehicles x 2
        # Compute norms with thresholding
        vel_norm = np.linalg.norm(vel_t, axis=1, keepdims=True)
        vectors_norm = np.linalg.norm(vectors_to_spots, axis=2, keepdims=True)
        epsilon = 1e-6
        vel_norm = np.maximum(vel_norm, epsilon)
        vectors_norm = np.maximum(vectors_norm, epsilon)
        # Normalize the vectors
        vel_normalized = vel_t / vel_norm
        vectors_normalized = vectors_to_spots / vectors_norm
        # Compute cosine similarity
        dot_prod = np.einsum('ij,mij->mi', vel_t, vectors_to_spots)
        cosine_similarity = np.einsum('ij,mij->mi', vel_normalized, vectors_normalized)
        zero_spot_veh = np.logical_or(vel_norm[:, 0] <= epsilon, vectors_norm[:, :, 0] <= epsilon)
        spot_ind, veh_ind = np.where(zero_spot_veh)
        # Convert to cosine distance (1 - cosine similarity)
        cosine_distance = 1 - cosine_similarity # (n_spots, n_vehicles)
        cosine_distance[spot_ind, veh_ind] = 0.0
        dot_prod[spot_ind, veh_ind] = 100000.0

        cos_theta = np.cos(yaw_t)
        sin_theta = np.sin(yaw_t)
        rotation_matrices = np.stack([
            np.stack([cos_theta, -sin_theta], axis=-1),
            np.stack([sin_theta, cos_theta], axis=-1)
        ], axis=-2)
        Sigma_t = np.array([rotation_matrices[i] @ (Sigma_t[i] + Q) @ rotation_matrices[i].T  for i in range(Sigma_t.shape[0])])  # Update covariances

        rho = 0.0

        Sigma_vel_t = (2*Sigma_t + Q - 2*np.array([[rho, 0.0],
                                                   [0.0, rho]]))/MOTION_RESOLUTION**2
        veh_vel_uncertainty = np.trace(Sigma_vel_t, axis1=1, axis2=2) # (n_vehicles,)
        # Compute probability of being inside parking spot for each vehicle and each spot (n_spots x n_vehicles)
        prob_t_dist = np.array([[mahalanobis_distance(mu_t[i], Sigma_t[i], park_spots_xy[j], Car_obj) for i in range(n_vehicles)] for j in range(n_spots)])
        prob_t =prob_t_dist*(1/(1+np.exp(-0.001*dot_prod/veh_vel_uncertainty[np.newaxis, :])))# including velocity information
        # prob_t = prob_t_dist

        prob_t_new = 1 - np.prod(1 - prob_t, axis=1)  # Probability that at least one vehicle occupies each spot using only dynamic obstacles
        alpha = 0.95 # filter: exponential distribution for departure rate, poisson for arrival rate
        P_O[t] = alpha * P_O[t - 1] + prob_t_new * (1 - P_O[t - 1])  # At least one vehicle per spot
        # P_O[t] = prob_t_new

        P_O_d[t] = 1-(alpha * (1-P_O_d[t - 1]) + (1-prob_t_new) * (P_O_d[t - 1]))

        # P_O[t] = prob_t1
        # P_enter_t = norm.cdf(d_t)  # Probability vehicle i enters spot j
        #
        # # Update occupancy probability recursively for all vehicles and spots
        # P_O[t] = alpha * P_O[t - 1] + (1 - alpha) * (
        #             1 - np.prod(1 - P_enter_t, axis=0))  # At least one vehicle per spot

    # P_O_result[:, 0] =  P_O[:, 0]
    # P_O_result[:, 1] = P_O_d[:, 1]

    P_O_result[:, occ_spots] =  P_O_d[:, occ_spots]
    P_O_result[:, vac_spots] = P_O[:, vac_spots]

    return P_O_result

type='big_lot'

home_path = os.path.abspath(os.getcwd())

## Load the configuration files
config_path = home_path + '/Config/'

with open(config_path + 'config_planner.json') as f:
    config_planner = json.load(f)

with open(config_path + 'config_map.json') as f:
    config_map = json.load(f)

Car_obj = Car_class(config_planner)
x_min, x_max, y_min, y_max, p_w, p_l, l_w, n_r, n_s, n_s1, obstacleX, obstacleY, s = map_lot(type, config_map, Car_obj)

# Example parameters
T = 10  # Number of time steps
n_spots = 2  # Number of parking spots

ox = obstacleX
oy = obstacleY

# Set Initial parameters
# start = [-5.0, 4.35, 0]
wb_2 = Car_obj.wheelBase / 2
start = [s[0] + wb_2 * np.cos(s[2]), s[1] + wb_2 * np.sin(s[2]), s[2]]  # transforming to center of vehicle
# goal = [0.0, 0.0, np.deg2rad(-90.0)]
# goal = [0, 0, np.deg2rad(-90.0)]
# goal = [0, 0, np.deg2rad(90)]
park_spots = [5, 9]
# park_spots_xy: list of centers of park_spots [x, y, 1 if left to center line and 0 if right to center line]
park_spots_xy = np.array([np.array([x_min + (1 + (i // n_s)) * l_w + (i // n_s1) * p_l + p_l / 2,
                           y_min + l_w + (i % n_s1) * p_w + p_w / 2, bool(i % n_s <= n_s1 - 1)])
                 for i in park_spots])
goal_park_spots = []  # park_spot i ->  goal yaw = 0.0 if 0, and np.pi if 1 -> (x, y, yaw) of goal
for spot_xy in park_spots_xy:
    # transforming center of parking spot to rear axle of vehicle (goal x, y) with appropriate goal yaw
    goal1 = np.array([spot_xy[0] - Car_obj.length / 2 + Car_obj.axleToBack, spot_xy[1], 0.0])
    goal2 = np.array([spot_xy[0] + Car_obj.length / 2 - Car_obj.axleToBack, spot_xy[1], np.pi])
    goal_spot_xy = [goal1, goal2]
    goal_plot = goal1
    goal_park_spots.append(goal_spot_xy)

goal_park_spots = list(chain.from_iterable(goal_park_spots))
# transforming to center of vehicle
g_list = [[g[0] + wb_2 * np.cos(g[2]), g[1] + wb_2 * np.sin(g[2]), g[2]] for g in goal_park_spots]

n_vehicles = 1
# np.array([x_min + l_w + 2 * p_l + 3*l_w / 4, y_min + l_w + n_s1 * p_w, np.deg2rad(90.0)]) # right lane outside parking row
# np.array([x_min + 1*l_w + 2 * p_l + 1*l_w / 4, y_min + l_w + (n_s1-1) * p_w, np.deg2rad(-90.0)]) # left lane outside parking row
dynamic_veh_0 = np.array([np.array([x_min + 1*l_w + 2 * p_l + 1*l_w / 4, y_min + l_w + (n_s1-1) * p_w, np.deg2rad(-90.0)])
                          ])
dynamic_veh_0 = dynamic_veh_0[:n_vehicles]
# dynamic_veh_0 = np.array([np.array([x_min + l_w + 2 * p_l + l_w / 4, y_min + l_w + n_s1 * p_w, np.deg2rad(-90.0)])]) # parking in
Sigma_0 = np.array([[[0.5*Car_obj.length, 0], [0, 0.5*Car_obj.width]],
                    [[0.5*Car_obj.length, 0], [0, 0.5*Car_obj.width]],
                    [[0.005*Car_obj.length, 0], [0, 0.005*Car_obj.width]]])  # Covariance for each vehicle
Sigma_0 = Sigma_0[:n_vehicles]
dynamic_veh_vel = np.array([np.array([0.0, -0.5, 0.0]),
                            np.array([-0.01, 0.0, 0.0]),
                            np.array([0.0, -0.001, 0.0])])
dynamic_veh_vel = dynamic_veh_vel[:n_vehicles]
dynamic_veh_parking = [0, 0, 0] # 1 is parking in, 2 is getting out, 0 is cruising
dynamic_veh_parking = dynamic_veh_parking[:n_vehicles]
length_preds = T+1
dynamic_veh_path = []
for veh_i, veh_parking in enumerate(dynamic_veh_parking):
    if veh_parking==1:
        ## dynamic_veh to spot
        path_veh = hybrid_a_star_planning(dynamic_veh_0[0], g_list[2], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        if length_preds > len(path_veh.x_list):
            extra_time_steps = int(length_preds - len(path_veh.x_list))
            path_veh_n = np.array([path_veh.x_list, path_veh.y_list, path_veh.yaw_list]).T
            last_state = path_veh_n[-1]
            repeat_veh = np.repeat(last_state.reshape((1, -1)), repeats=extra_time_steps, axis=0)
            path_veh_n = np.vstack((path_veh_n, repeat_veh))
        else:
            path_veh_n = np.array([path_veh.x_list, path_veh.y_list, path_veh.yaw_list]).T
            path_veh_n = path_veh_n[:length_preds]

    elif veh_parking==2:
        ## dynamic_veh out of spot
        path_veh = hybrid_a_star_planning(g_list[3], dynamic_veh_0[0], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
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
        veh_vel = dynamic_veh_vel[veh_i]
        veh_0 = dynamic_veh_0[veh_i]
        # straight_goal = dynamic_veh_0[0] + np.array([0.0, -15.0, 0.0])
        path_veh_n = np.array([veh_0 + i * veh_vel for i in range(length_preds)])

    dynamic_veh_path.append(path_veh_n)

dynamic_veh_path=np.array(dynamic_veh_path)
skip = 2 # wrt MOTION_RESOLUTION
T_prob  = int(T/skip)
dynamic_veh_path = dynamic_veh_path[:, 0:dynamic_veh_path.shape[1]:skip , :]
Q = np.array([[0.5, 0], [0, 0.5]])  # Process noise (uncertainty growth)

fig, ax = plt.subplots(figsize=(10, 8))
## Init plot
i_x, i_y, i_yaw = start[0], start[1], start[2]
ax.plot(ox, oy, ".k")
ax.set_xlim(x_min - 1, x_max + 1)
ax.set_ylim(y_min - 1, y_max + 1)
plot_car(i_x, i_y, i_yaw, ax)
time_dynamic = 0
angles = np.linspace(0, 2*np.pi, 100)
Sigma_t = Sigma_0
# plot other cars
for veh_i in range(len(dynamic_veh_path)):
    for time_dynamic in range(dynamic_veh_path[veh_i].shape[0]):
        p_yaw = dynamic_veh_path[veh_i, time_dynamic, 2]
        p_x = dynamic_veh_path[veh_i, time_dynamic, 0]
        p_y = dynamic_veh_path[veh_i, time_dynamic, 1]
        plot_other_car(p_x, p_y, p_yaw, ax)

        if time_dynamic == 0 or time_dynamic == dynamic_veh_path[veh_i].shape[0]-1:

            a=np.sqrt(2*(Sigma_t[veh_i, 0, 0] + Q[0, 0]))       #radius on the x-axis
            b=np.sqrt(2*(Sigma_t[veh_i, 1, 1] + Q[1, 1]))       #radius on the y-axis

            x = a * np.cos(angles)
            y = b * np.sin(angles)

            # Rotation matrix
            R = np.array([[np.cos(p_yaw), -np.sin(p_yaw)],
                          [np.sin(p_yaw), np.cos(p_yaw)]])

            # Apply rotation
            rotated_points = R @ np.vstack((x, y))

            # Shift to center
            x_rotated = rotated_points[0, :] + p_x
            y_rotated = rotated_points[1, :] + p_y

            ax.plot(x_rotated, y_rotated, color='red', alpha=0.5)
            ax.plot(p_x, p_y, linestyle='', marker='o', color='red')  # rotated ellipse

        Sigma_t[veh_i] = Sigma_t[veh_i] + Q

# Compute occupancy probabilities for multiple vehicles and spots
time_all = []
for i in range(100):
    start_t = time.time()
    P_O = occupancy_probability_multiple_spots(T_prob, dynamic_veh_path, Sigma_0, Q, park_spots_xy, Car_obj)
    time_all.append(time.time() - start_t)

fig_p, ax_p = plt.subplots()
for i in range(P_O.shape[1]):
    label_i = 'Spot' + str(i+1)
    ax_p.plot(np.arange(P_O.shape[0]), P_O[:, i], marker='o', markersize=12, label=label_i)
    ax_p.tick_params(axis='both', which='major', labelsize=20)
    ax_p.set_xlabel("Time (s)", fontsize=20)
    ax_p.set_ylabel("Occupancy Probability", fontsize=20)
    ax_p.set_yticks(np.arange(0, 1, 0.1))
    ax_p.grid(True)

ax_p.legend(fontsize=24)

# Print results
for t in range(T_prob + 1):
    print(f"Time {int(t)}:")
    for j in range(n_spots):
        print(f"  Spot {j + 1}: P(O_t) = {P_O[t, j]:.3f}")

avg_time = np.mean(time_all[1:])
std_time = np.std(time_all[1:])
print("Avg Comp time: ", avg_time)
print("STD Comp time: ", std_time)

ax.axis('equal')
plt.show()