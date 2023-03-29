import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatialmath import SE3, SO3
from PIL import Image
from tqdm import tqdm
from debug_helpers import *

# traj_dir = "/home/mokhtars/Documents/bc_network/collected_data/traj0"
# ee_info = []
# pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)

# for i in range(len(pose_file.item().keys())):
#     ee_info.append(pose_file.item()[i]["ee_pose"])

# orientations = np.array([pose.R for pose in ee_info])
# positions = [] 
# diffs = []
# diffs2 = []
# diffs_rot = []
# for i in range(len(ee_info)-1):
#     positions.append(ee_info[i].t)
#     orientation1 = orientations[i]
#     orientation1 = SO3(orientation1)
#     euler_angles1 = orientation1.eul()

#     orientation2 = orientations[i+1]
#     orientation2 = SO3(orientation2)
#     euler_angles2 = orientation2.eul()

#     diffs.append(orientations[i+1] @  orientations[i].T)
#     orientation3 = SO3(diffs[i])
#     euler_angles3 = orientation3.eul()
#     diffs_rot.append(euler_angles3)

#     diffs2.append(euler_angles2 - euler_angles1)



# def pos_rotation_visualization(ee_info):
#     # create a 3D plot
#     fig = plt.figure(figsize=(15, 15))
#     ax = fig.add_subplot(111, projection='3d')

#     # plot each pose as a set of arrows
#     for pose in ee_info:
#         # extract the position and rotation from the pose matrix
#         position = pose.t
#         rotation = pose.R

#         # calculate the endpoints of the arrows based on the position and rotation
#         arrow_x = position + 0.01 * np.dot(rotation, np.array([1, 0, 0]))
#         arrow_y = position + 0.01 * np.dot(rotation, np.array([0, 1, 0]))
#         arrow_z = position + 0.01 * np.dot(rotation, np.array([0, 0, 1]))

#         # plot the arrows
#         ax.quiver(position[0], position[1], position[2], arrow_x[0]-position[0], arrow_x[1]-position[1], arrow_x[2]-position[2], length=0.02, normalize=True, color='r')
#         ax.quiver(position[0], position[1], position[2], arrow_y[0]-position[0], arrow_y[1]-position[1], arrow_y[2]-position[2], length=0.02, normalize=True, color='g')
#         ax.quiver(position[0], position[1], position[2], arrow_z[0]-position[0], arrow_z[1]-position[1], arrow_z[2]-position[2], length=0.02, normalize=True, color='b')

#     # set the limits of the plot to show all the arrows
#     ax.set_xlim([min([pose.A[0, 3] for pose in ee_info])-0.1, max([pose.A[0, 3] for pose in ee_info])+0.1])
#     ax.set_ylim([min([pose.A[1, 3] for pose in ee_info])-0.1, max([pose.A[1, 3] for pose in ee_info])+0.1])
#     ax.set_zlim([min([pose.A[2, 3] for pose in ee_info])-0.1, max([pose.A[2, 3] for pose in ee_info])+0.1])

#     # add labels and show the plot
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()

# # pos_rotation_visualization(ee_info)



# def plt2arr(fig, draw=True):
#     """
#     need to draw if figure is not drawn yet
#     """
#     if draw:
#         fig.canvas.draw()
#     rgba_buf = fig.canvas.buffer_rgba()
#     (w,h) = fig.canvas.get_width_height()
#     rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))

#     return rgba_arr


# def gif_rot_euler_change(positions, rot_diffs, euler_diffs):
#     frames = []
#     euler_angles = euler_diffs
#     rot_mats = rot_diffs
#     positions = [0,0,0]
#     for idx in tqdm(range(len(diffs))):
#         euler_angle = euler_angles[idx]
#         rot_mat = rot_mats[idx]
#         # Create X, Y, and Z unit vectors in the world coordinate frame
#         X = np.array([1, 0, 0])
#         Y = np.array([0, 1, 0])
#         Z = np.array([0, 0, 1])
#         # Rotate the X, Y, and Z unit vectors by the Euler angles
#         R_x = np.array([[1, 0, 0],
#                         [0, np.cos(euler_angle[0]), -np.sin(euler_angle[0])],
#                         [0, np.sin(euler_angle[0]), np.cos(euler_angle[0])]])

#         R_y = np.array([[np.cos(euler_angle[1]), 0, np.sin(euler_angle[1])],
#                         [0, 1, 0],
#                         [-np.sin(euler_angle[1]), 0, np.cos(euler_angle[1])]])

#         R_z = np.array([[np.cos(euler_angle[2]), -np.sin(euler_angle[2]), 0],
#                         [np.sin(euler_angle[2]), np.cos(euler_angle[2]), 0],
#                         [0, 0, 1]])

#         X_eul = R_z @ R_y @ R_x @ X
#         Y_eul = R_z @ R_y @ R_x @ Y
#         Z_eul = R_z @ R_y @ R_x @ Z

#         X_rot = rot_mat @ X
#         Y_rot = rot_mat @ Y
#         Z_rot = rot_mat @ Z

#         # Create a 3D plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')

#         # Plot the position vector as a red point
#         ax.scatter(positions[0], positions[1], positions[2], color='black')

#         # Plot the X, Y, and Z axes for the first orientation
#         ax.quiver(positions[0], positions[1], positions[2], X_eul[0]/20, X_eul[1]/20, X_eul[2]/20, color="#FF0000", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Y_eul[0]/20, Y_eul[1]/20, Y_eul[2]/20, color="#008000", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Z_eul[0]/20, Z_eul[1]/20, Z_eul[2]/20, color="#0e2ced", arrow_length_ratio=0.1)

#         # Plot the X, Y, and Z axes for the second orientation
#         ax.quiver(positions[0], positions[1], positions[2], X_rot[0]/20, X_rot[1]/20, X_rot[2]/20, color="#FF5733", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Y_rot[0]/20, Y_rot[1]/20, Y_rot[2]/20, color="#98FB98", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Z_rot[0]/20, Z_rot[1]/20, Z_rot[2]/20, color="#0ae1f5", arrow_length_ratio=0.1)
#         ax.view_init(elev=15, azim=20)

#         ax.set_title(f'{str(idx+1)} Euler darker')

#         frames.append(plt2arr(fig, True))
#         plt.close(fig)

#     gif = [Image.fromarray(img) for img in frames]
#     gif[0].save("pos_ori_action.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)




# def gif_euler_euler_change(positions, net_diffs, euler_diffs):
#     frames = []
#     euler_angles = euler_diffs
#     net_angles = net_diffs
#     positions = [0,0,0]
#     for idx in tqdm(range(len(diffs)-2)):
#         euler_angle = euler_angles[idx]
#         net_angle = net_angles[idx]
#         # Create X, Y, and Z unit vectors in the world coordinate frame
#         X = np.array([1, 0, 0])
#         Y = np.array([0, 1, 0])
#         Z = np.array([0, 0, 1])
#         # Rotate the X, Y, and Z unit vectors by the Euler angles
#         R_x = np.array([[1, 0, 0],
#                         [0, np.cos(euler_angle[0]), -np.sin(euler_angle[0])],
#                         [0, np.sin(euler_angle[0]), np.cos(euler_angle[0])]])

#         R_y = np.array([[np.cos(euler_angle[1]), 0, np.sin(euler_angle[1])],
#                         [0, 1, 0],
#                         [-np.sin(euler_angle[1]), 0, np.cos(euler_angle[1])]])

#         R_z = np.array([[np.cos(euler_angle[2]), -np.sin(euler_angle[2]), 0],
#                         [np.sin(euler_angle[2]), np.cos(euler_angle[2]), 0],
#                         [0, 0, 1]])

#         X_eul = R_z @ R_y @ R_x @ X
#         Y_eul = R_z @ R_y @ R_x @ Y
#         Z_eul = R_z @ R_y @ R_x @ Z

#         R_x2 = np.array([[1, 0, 0],
#                         [0, np.cos(net_angle[0, 0]), -np.sin(net_angle[0, 0])],
#                         [0, np.sin(net_angle[0, 0]), np.cos(net_angle[0, 0])]])

#         R_y2 = np.array([[np.cos(net_angle[0, 1]), 0, np.sin(net_angle[0, 1])],
#                         [0, 1, 0],
#                         [-np.sin(net_angle[0, 1]), 0, np.cos(net_angle[0, 1])]])

#         R_z2 = np.array([[np.cos(net_angle[0, 2]), -np.sin(net_angle[0, 2]), 0],
#                         [np.sin(net_angle[0, 2]), np.cos(net_angle[0, 2]), 0],
#                         [0, 0, 1]])

#         X_net = R_z2 @ R_y2 @ R_x2 @ X
#         Y_net = R_z2 @ R_y2 @ R_x2 @ Y
#         Z_net = R_z2 @ R_y2 @ R_x2 @ Z

#         # Create a 3D plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')

#         # Plot the position vector as a red point
#         ax.scatter(positions[0], positions[1], positions[2], color='black')

#         # Plot the X, Y, and Z axes for the initial orientation as blue arrows
#         ax.quiver(positions[0], positions[1], positions[2], X_eul[0]/20, X_eul[1]/20, X_eul[2]/20, color="#FF0000", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Y_eul[0]/20, Y_eul[1]/20, Y_eul[2]/20, color="#008000", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Z_eul[0]/20, Z_eul[1]/20, Z_eul[2]/20, color="#0e2ced", arrow_length_ratio=0.1)

#         # Plot the X, Y, and Z axes for the first rotated orientation as green arrows
#         ax.quiver(positions[0], positions[1], positions[2], X_net[0]/20, X_net[1]/20, X_net[2]/20, color="#FF5733", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Y_net[0]/20, Y_net[1]/20, Y_net[2]/20, color="#98FB98", arrow_length_ratio=0.1)
#         ax.quiver(positions[0], positions[1], positions[2], Z_net[0]/20, Z_net[1]/20, Z_net[2]/20, color="#0ae1f5", arrow_length_ratio=0.1)
#         ax.view_init(elev=15, azim=20)

#         ax.set_title(f'{str(idx+1)} Network Euler Lighter')

#         frames.append(plt2arr(fig, True))
#         plt.close(fig)

#     gif = [Image.fromarray(img) for img in frames]
#     gif[0].save("Euler_network.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)


# image_data, joint_data, action = experience_collector(traj_dir)
# action_diffs = action[:,:,3:-1]
# # gif_euler_euler_change(positions, action_diffs, diffs2)
# # gif_rot_euler_change(positions, diffs, diffs2)
# print("Maximum element of action change: ", torch.max(action_diffs))
# print("Minimum element of action change: ", torch.min(action_diffs))

# def output_processing(current_pose, next_action):
#     rot = current_pose.R
#     current_ori = R.from_matrix(rot)
#     current_ori = current_ori.as_euler("xyz")
#     current_pos = current_pose.t
#     diff_ori = np.array([next_action[0, 3], next_action[0, 4], next_action[0, 5]])
#     target_ori = [current_ori[0] + diff_ori[0], current_ori[1] + diff_ori[1], current_ori[2] + diff_ori[2]]
#     target_pos = [next_action[0, 0] + current_pos[0], next_action[0, 1] + current_pos[1], next_action[0, 2] + current_pos[2]] 
#     r = R.from_euler("xyz", [target_ori[0], target_ori[1], target_ori[2]])
#     target_ori = sm.SO3(trnorm(np.array(r.as_matrix())))
#     target_pose = sm.SE3.Rt(target_ori, target_pos)
#     return target_pose
    
# ee_processed = [ee_info[0]]
# for i in range(action.shape[0]):
#     ee_processed.append(output_processing(ee_info[i], action[i]))

# # pos_rotation_visualization(ee_processed)



data_dir = "/home/mokhtars/Documents/bc_network"
target_file = np.load(data_dir + "/target_ee_pose.npy", allow_pickle=True)
actual_file = np.load(data_dir + "/actual_ee_pose.npy", allow_pickle=True)
target_data = target_file.item()
actual_data = actual_file.item()
print()
# Extract the positions from the dictionaries
target_positions = np.array([target_data[i].t for i in range(len(target_data))])
actual_positions = np.array([actual_data[i].t for i in range(len(actual_data))])

# Plot the positions as points in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(target_positions[:,0], target_positions[:,1], target_positions[:,2], color='b', label='Target')
ax.scatter(actual_positions[:,0], actual_positions[:,1], actual_positions[:,2], color='r', label='Actual')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
plt.show()