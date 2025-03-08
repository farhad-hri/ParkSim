import numpy as np
from scipy.spatial import distance

from parksim.path_planner.hybrid_astar.hybrid_a_star_parallel import map_lot, Car_class, hybrid_a_star_planning, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, MOTION_RESOLUTION
from parksim.path_planner.hybrid_astar.car import plot_car, plot_other_car

def mahalanobis_distance(mu, Sigma, park_spot, Car_obj, ped_flag):
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

    if ped_flag:
        d_min = 2*np.linalg.norm(park_spot[:2] - mu[:2])
    else:
        d_min = distance.mahalanobis(park_spot[:2], mu[:2], Sigma)

    alpha_g = 350
    prob = (alpha_g+1)/(alpha_g+np.exp(d_min))
    # prob = (norm.cdf(d_x_max) - norm.cdf(d_x_min))*(norm.cdf(d_y_max) - norm.cdf(d_y_min))

    return prob

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
    P_O_vacant = np.zeros((T + 1, len(vac_spots)))  # Occupancy probability for each time step and parking spot
    P_O_occ = np.zeros((T + 1, len(occ_spots_veh) + len(occ_spots_ped)))  # Occupancy probability for each time step and parking spot, focussed on departing agent

    # Initialize with multiple vehicles' initial probability of occupying the spots
    yaw_t = dynamic_veh_path_t[:, 0, 2]
    cos_theta = np.cos(yaw_t)
    sin_theta = np.sin(yaw_t)
    rotation_matrices = np.stack([
        np.stack([cos_theta, -sin_theta], axis=-1),
        np.stack([sin_theta, cos_theta], axis=-1)
    ], axis=-2)
    Q_ped = 1*Q
    Sigma_t = np.array([rotation_matrices[i] @ (Sigma_0[i] + Q) @ rotation_matrices[i].T  for i in range(Sigma_0.shape[0])])  # Update covariances for dynamic veh
    Sigma_t_ped = np.array([Sigma_0_ped[i] + Q_ped for i in range(Sigma_0_ped.shape[0])]) # Update covariances for ped
    n_vehs  = Sigma_t.shape[0]

    Sigma_t_all = np.vstack((Sigma_t, Sigma_t_ped))
    n_agents = Sigma_t_all.shape[0]
    park_spots_xy = np.vstack((center_spots[vac_spots], center_spots[occ_spots_veh], center_spots[occ_spots_ped]))
    n_spots = park_spots_xy.shape[0]

    agent_path = np.vstack((dynamic_veh_path_t[:, :, :2], ped_path_t))

    prob_t_dist = np.array([[mahalanobis_distance(agent_path[i, 0], Sigma_t_all[i], park_spots_xy[j], Car_obj, int(i/n_vehs)) for i in range(n_agents)] for j in range(n_spots)]) # (n_spots, n_agents)

    Sigma_t = Sigma_0
    Sigma_t_ped = Sigma_0_ped

    dynamic_veh_vel = np.diff(dynamic_veh_path_t[:, :, :2], axis=1) / MOTION_RESOLUTION
    ped_vel = np.diff(ped_path_t[:, :, :2], axis=1) / MOTION_RESOLUTION

    ## velocity information
    mu_t = np.vstack((dynamic_veh_path_t[:, 0, :2], ped_path_t[:, 0, :2]))
    vel_t = np.vstack((dynamic_veh_vel[:, 0, :2], ped_vel[:, 0, :2]))  # n_agents x 2
    vectors_to_spots = park_spots_xy[:, np.newaxis, :2] - mu_t[np.newaxis, :, :2]  # n_spots x n_agents x 2
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
    zero_spot_agent = np.logical_or(vel_norm[:, 0] <= epsilon, vectors_norm[:, :, 0] <= epsilon)
    spot_ind, agent_ind = np.where(zero_spot_agent)
    # Convert to cosine distance (1 - cosine similarity)
    cosine_distance = 1 - cosine_similarity  # (n_spots, n_vehicles)
    cosine_distance[spot_ind, agent_ind] = 0.0
    dot_prod[spot_ind, agent_ind] = 100000.0

    rho = 0.8

    Sigma_vel_t = (2 * Sigma_t_all + Q - 2 * np.array([[rho, 0.0],
                                                       [0.0, rho]])) / MOTION_RESOLUTION ** 2
    vel_uncertainty = np.trace(Sigma_vel_t, axis1=1, axis2=2)  # (n_agents,)
    prob_t = prob_t_dist * (1 / (1 + np.exp(-0.001 * dot_prod / vel_uncertainty[np.newaxis, :])))  # including velocity information
    prob_t_new = 1 - np.prod(1 - prob_t, axis=1)  # Probability that at least one agent occupies each spot using only dynamic agents
    P_O_vacant[0] = prob_t_new[:len(vac_spots)]
    P_O_occ[0] = prob_t_new[len(vac_spots):len(vac_spots)+len(occ_spots_veh)+len(occ_spots_ped)]

    # Iterate through time steps t = 1 to T
    for k in range(1, T + 1):
        # Propagate mean and covariance for each agent
        mu_t = np.vstack((dynamic_veh_path_t[:, k, :2], ped_path_t[:, k, :2]))   # Update positions
        yaw_t = dynamic_veh_path_t[:, k, 2]

        vel_t = np.vstack((dynamic_veh_vel[:, min(k, T-1), :2], ped_vel[:, min(k, T-1), :2])) # n_agents x 2
        vectors_to_spots = park_spots_xy[:, np.newaxis, :2] - mu_t[np.newaxis, :, :2] # n_spots x n_agents x 2
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
        zero_spot_agent = np.logical_or(vel_norm[:, 0] <= epsilon, vectors_norm[:, :, 0] <= epsilon)
        spot_ind, agent_ind = np.where(zero_spot_agent)
        # Convert to cosine distance (1 - cosine similarity)
        cosine_distance = 1 - cosine_similarity # (n_spots, n_vehicles)
        cosine_distance[spot_ind, agent_ind] = 0.0
        dot_prod[spot_ind, agent_ind] = 100000.0

        cos_theta = np.cos(yaw_t)
        sin_theta = np.sin(yaw_t)
        rotation_matrices = np.stack([
            np.stack([cos_theta, -sin_theta], axis=-1),
            np.stack([sin_theta, cos_theta], axis=-1)
        ], axis=-2)

        Sigma_t = np.array([rotation_matrices[i] @ (Sigma_t[i] + Q) @ rotation_matrices[i].T for i in range(Sigma_t.shape[0])])  # Update covariances for dynamic veh
        Sigma_t_ped = np.array([Sigma_t_ped[i] + Q_ped for i in range(Sigma_t_ped.shape[0])])  # Update covariances for ped
        Sigma_t_all = np.vstack((Sigma_t, Sigma_t_ped))

        rho = 0.0

        Sigma_vel_t = (2*Sigma_t_all + Q - 2*np.array([[rho, 0.0],
                                                   [0.0, rho]]))/MOTION_RESOLUTION**2
        vel_uncertainty = np.trace(Sigma_vel_t, axis1=1, axis2=2) # (n_agents,)
        # Compute probability of being inside parking spot for each agent and each spot (n_spots x n_agents)
        prob_t_dist = np.array([[mahalanobis_distance(mu_t[i], Sigma_t_all[i], park_spots_xy[j], Car_obj, int(i/n_vehs)) for i in range(n_agents)] for j in range(n_spots)])
        prob_t = prob_t_dist*(1/(1+np.exp(-0.001*dot_prod/vel_uncertainty[np.newaxis, :])))# including velocity information
        # prob_t = prob_t_dist

        prob_t_new = 1 - np.prod(1 - prob_t, axis=1)  # Probability that at least one agent occupies each spot using only dynamic agents
        alpha = 0.95 # filter: exponential distribution for departure rate, poisson for arrival rate
        P_O_vacant[k] = alpha * P_O_vacant[k - 1] + prob_t_new[:len(vac_spots)] * (1 - P_O_vacant[k - 1])

        # P_O_vacant[k] = prob_t_new[:len(vac_spots)]
        # P_O_occ[k] = prob_t_new[len(vac_spots):len(vac_spots)+len(occ_spots_veh)+len(occ_spots_ped)]

        P_O_occ[k] = 1-(alpha * (1-P_O_occ[k - 1]) + (1-prob_t_new[len(vac_spots):len(vac_spots)+len(occ_spots_veh)+len(occ_spots_ped)]) * (P_O_occ[k - 1]))

    return P_O_vacant, P_O_occ
