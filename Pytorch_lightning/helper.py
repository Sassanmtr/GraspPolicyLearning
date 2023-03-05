import os

import numpy as np

# Example rotation matrix
R = np.array([[0.40673664, -0.91354546, 0.04055254],
              [0.91354546, 0.40673664, 0.00174532],
              [-0.04055254, 0.00174532, 0.99916584]])

# data_dir = "/home/mokhtars/Documents/bc_network/bc_network/trajecories/validation"
# folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
# folders.sort()
# first_folder = folders[0] if folders else None
# print(first_folder)
# index = int(first_folder[4:])
# print(index)
# print(type(index))

def rotation_angles(matrix):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    theta1 = np.arctan(-r23 / r33)
    theta2 = np.arctan(r13 * np.cos(theta1) / r33)
    theta3 = np.arctan(-r12 / r11)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return np.array((theta1, theta2, theta3))

a = rotation_angles(np.array([[1,1,1],[0,1,0],[0,0,1]]))