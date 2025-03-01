import numpy as np
from scipy.stats import norm
import os
import json
from itertools import chain
import matplotlib.pyplot as plt

from parksim.path_planner.hybrid_astar.hybrid_a_star_parallel import map_lot, Car_class, hybrid_a_star_planning, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION
from parksim.path_planner.hybrid_astar.car import plot_car, plot_other_car

def mahalanobis_distance(mu, Sigma, park_spot, Car_obj):
    """
    Computes the Mahalanobis distance from the mean position `mu` to the nearest
    boundary of the rectangular parking spot.

    Parameters:
    - mu: pose of vehicle (3, )
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

    prob = (norm.cdf(d_x_max) - norm.cdf(d_x_min))*(norm.cdf(d_y_max) - norm.cdf(d_y_min))

    return prob

def occupancy_probability_multiple_spots(T, dynamic_veh_path, Sigma_0, Q, park_spots_xy, Car_obj):
    """
    Computes the recursive occupancy probability P(O_t) for multiple vehicles and multiple parking spots.

    Parameters:
    - T: Total number of time steps
    - dynamic_veh_path: path of dynamic vehicle's mean (n_vehicles x 3 array)
    - Sigma_0: Initial covariance matrix of the vehicles' positions (n_vehicles x 2 x 2 array)
    - Q: Process noise covariance matrix (2 x 2)
    - park_spots_xy: List of centers(x, y) of parking spots and whether parking spot is left (1) or right (0) to center lane  (n_spots x 3)
    - Car_obj: object of Car_class that stores the dimensions of car

    Returns:
    - P_O: Occupancy probability over time for each parking spot.
    """
    n_spots = park_spots_xy.shape[0]
    n_vehicles = dynamic_veh_path.shape[0]
    P_O = np.zeros((T + 1, n_spots))  # Occupancy probability for each time step and parking spot

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

    mu_t = dynamic_veh_path[:, 0]
    Sigma_t = Sigma_0

    # Iterate through time steps t = 1 to T
    for t in range(1, T + 1):
        # Propagate mean and covariance for each vehicle
        mu_t = dynamic_veh_path[:, t, :2]  # Update positions (vectorized)
        yaw_t = dynamic_veh_path[:, t, 2]
        cos_theta = np.cos(yaw_t)
        sin_theta = np.sin(yaw_t)
        rotation_matrices = np.stack([
            np.stack([cos_theta, -sin_theta], axis=-1),
            np.stack([sin_theta, cos_theta], axis=-1)
        ], axis=-2)
        Sigma_t = np.array([rotation_matrices[i] @ (Sigma_t[i] + Q) @ rotation_matrices[i].T  for i in range(Sigma_t.shape[0])])  # Update covariances

        # Compute probability of being inside parking spot for each vehicle and each spot
        prob_t = np.array([[mahalanobis_distance(mu_t[i], Sigma_t[i], park_spots_xy[j], Car_obj) for i in range(n_vehicles)] for j in range(n_spots)])

        P_O[t] = 1 - np.prod(1 - prob_t, axis=1) # Probability that at least one vehicle occupies each spot
        # P_O[t] = prob_t1
        # P_enter_t = norm.cdf(d_t)  # Probability vehicle i enters spot j
        #
        # # Update occupancy probability recursively for all vehicles and spots
        # P_O[t] = alpha * P_O[t - 1] + (1 - alpha) * (
        #             1 - np.prod(1 - P_enter_t, axis=0))  # At least one vehicle per spot

    return P_O

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

# Example parameters
T = 50  # Number of time steps
n_vehicles = 1  # Number of vehicles
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

dynamic_veh_0 = np.array([np.array([x_min + l_w + 2 * p_l + l_w / 4, y_min + l_w + n_s1 * p_w, np.deg2rad(-90.0)])])
Sigma_0 = np.array([[[1*Car_obj.length, 0], [0, 1*Car_obj.width]]])  # Covariance for each vehicle
dynamic_veh_vel = np.array([np.array([0.0, -0.5, 0.0])])
dynamic_veh_parking = [1]
length_preds = T+1
dynamic_veh_path = []
for veh_i, veh_parking in enumerate(dynamic_veh_parking):
    if veh_parking:
        ## dynamic_veh to spot
        path_veh = hybrid_a_star_planning(dynamic_veh_0[0], g_list[1], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        extra_time_steps = int(length_preds - len(path_veh.x_list))
        path_veh_n = np.array([path_veh.x_list, path_veh.y_list, path_veh.yaw_list]).T
        last_state = path_veh_n[-1]
        repeat_veh = np.repeat(last_state.reshape((1, -1)), repeats=extra_time_steps, axis=0)
        path_veh_n = np.vstack((path_veh_n, repeat_veh))
    else:
        ## dynamic_veh constant velocity
        veh_vel = dynamic_veh_vel[veh_i]
        veh_0 = dynamic_veh_0[veh_i]
        # straight_goal = dynamic_veh_0[0] + np.array([0.0, -15.0, 0.0])
        path_veh_n = np.array([veh_0 + i * veh_vel for i in range(length_preds)])

    dynamic_veh_path.append(path_veh_n)

dynamic_veh_path=np.array(dynamic_veh_path)
skip = 3
T = int(T/skip)
dynamic_veh_path = dynamic_veh_path[:, 0:dynamic_veh_path.shape[1]:skip , :]
Q = np.array([[0.2, 0], [0, 0.2]])  # Process noise (uncertainty growth)

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
P_O = occupancy_probability_multiple_spots(T, dynamic_veh_path, Sigma_0, Q, park_spots_xy, Car_obj)

# Print results
for t in range(T + 1):
    print(f"Time {t}:")
    for j in range(n_spots):
        print(f"  Spot {j + 1}: P(O_t) = {P_O[t, j]:.3f}")

ax.axis('equal')
plt.show()