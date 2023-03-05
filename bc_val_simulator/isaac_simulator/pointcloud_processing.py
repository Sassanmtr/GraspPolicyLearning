import numpy as np
import matplotlib.pyplot as plt


def get_obstacle_map(robot_interface, n_bins=24):
    """
    Returns two arrays, one containing the unit direction vector to the obstacle, and one the distance.
    The directions are discretized in n_bins, to reduce compute time in the optimization
    """
    base_r = 0.62
    front_pc, rear_pc = robot_interface.get_lidar_data()

    # Change reference from lidar frame to base frame. TODO: this should be the world frame!!!
    front_pc = np.array([front_lidar_to_base(point.T) for point in front_pc]).squeeze()
    rear_pc = np.array([rear_lidar_to_base(point.T) for point in rear_pc]).squeeze()

    # Concatenate Points
    pc = np.concatenate((front_pc, rear_pc))

    # Filter out far away points
    distances = np.linalg.norm(pc, axis=1)
    indeces = distances < base_r + 1.0  # radius of base + 1 meter
    pc = pc[indeces]
    distances = distances[indeces]
    angles = np.arctan2(pc[:, 1], pc[:, 0])
    if pc.size == 0:
        return pc, distances

    # Cluster Similar points together
    cluster_indeces = np.array([], dtype=np.int8)
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(angles, bins) - 1
    for i in range(n_bins):
        bin_i_idx = bin_idx == i
        if not any(bin_i_idx):
            continue
        masked_dist = np.ma.array(distances, mask=~bin_i_idx)
        idx = np.ma.argmin(masked_dist)
        cluster_indeces = np.append(cluster_indeces, idx)
    pc_out, dist_out = pc[cluster_indeces], distances[cluster_indeces]
    # Debug plot
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.cla()  # clear previous measurements
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.scatter(pc_out[:, 0], pc_out[:, 1])
    return pc_out, dist_out


def front_lidar_to_base(point):
    """
    Change of coordinate frame, only translation. (Also ignores z-value)
    """
    return [point[0] + 0.3932, point[1]]


def rear_lidar_to_base(point):
    """
    Change of coordinate frame, translation as well as 180 deg rotation (Also ignores z-value)
    """
    homog_transf = np.array([[-1, 0, -0.3932], [0, -1, 0], [0, 0, 1]])
    point_array = np.array([point[0], point[1], 1.0]).T
    result = homog_transf @ point_array
    return [result[0], result[1]]
