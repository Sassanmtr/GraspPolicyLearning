import os
import random
import numpy as np
import torch
from PIL import Image
import cv2
import h5py
from scipy.spatial.transform import Rotation as R
import spatialmath as sm
import roma
from spatialmath.base import trnorm


def mesh_data(mesh_dir):
    mesh = h5py.File(mesh_dir, "r")
    success_idcs = mesh["grasps"]["qualities"]["flex"]["object_in_gripper"][()]
    success_idcs == 1
    grasp_pts = mesh["grasps"]["transforms"][()]
    success_grasps = grasp_pts[success_idcs.nonzero()]
    obj_pos = mesh["object"]["com"][()]
    obj_scale = mesh["object"]["scale"][()]
    obj_mass = mesh["object"]["mass"][()]
    return success_grasps, obj_pos, obj_scale, obj_mass


def gripper_inital_point_selector():
    area = 2  # comment this line for random initial pose of gripper
    if area == 0:
        lower = np.array([1.05, 0.21, 1.19])
        upper = np.array([0.95, 0.18, 1.17])
        # y_rot = np.array([-33, -27])
        # theta = np.random.randint(low=y_rot[0], high=y_rot[1], size=1)
        # r = R.from_euler("xyz", [180, theta[0], 0], degrees=True)
        theta = -70
        r = R.from_euler("xyz", [180, theta, 0], degrees=True)
    elif area == 1:
        lower = np.array([0.2, 0.7, 0.9])
        upper = np.array([0.2, 0.7, 0.9])
        theta = 70
        r = R.from_euler("xyz", [180, theta, 90], degrees=True)
    elif area == 2:
        lower = np.array([1.0, 0.20, 1.30])
        upper = np.array([1.0, 0.20, 1.30])
        theta = -55
        r = R.from_euler("xyz", [180, theta, 0], degrees=True)
    else:
        lower = np.array([-0.5, 0.2, 0.9])
        upper = np.array([-0.5, 0.2, 0.9])
        theta = -70
        r = R.from_euler("xyz", [180, theta, 180], degrees=True)
    x = np.random.uniform(low=lower[0], high=upper[0], size=1)
    y = np.random.uniform(low=lower[1], high=upper[1], size=1)
    z = np.random.uniform(low=lower[2], high=upper[2], size=1)
    target_ori = sm.SO3(np.array(r.as_matrix()))
    target_pos = [x[0], y[0], z[0]]
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose


def output_processing(current_ee_pose, next_action):
    rot = current_ee_pose.R
    current_ori = mat2euler(rot)
    current_pos = current_ee_pose.t
    target_ori = [
        current_ori[0] + next_action[3],
        current_ori[1] + next_action[4],
        current_ori[2] + next_action[5],
    ]
    target_pos = [
        current_pos[0] + next_action[0],
        current_pos[1] + next_action[1],
        current_pos[2] + next_action[2],
    ]
    target_ori = sm.SO3(trnorm(euler2mat(target_ori)), check=False)
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose


def predict_input_processing(robot_interface, robot_sim, device):
    current_data = robot_interface.get_camera_data()
    rgb_ar = current_data["rgb"][:, :, 0:3]
    rgb_ar = torch.tensor(rgb_ar)
    depth_ar = current_data["depth"]
    depth_threshold = 7.0
    depth_ar[depth_ar > depth_threshold] = depth_threshold
    depth_ar = depth_ar.reshape(depth_ar.shape[0], depth_ar.shape[1], 1)
    depth_ar = torch.tensor(depth_ar)
    im_ar = torch.cat((rgb_ar, depth_ar), dim=-1)
    # print("Depth max: ", torch.max(depth_ar))
    im_ar = im_ar.permute(2, 0, 1)
    joint_ar = robot_sim.get_joint_positions()
    joint_ar = torch.tensor(joint_ar)
    return im_ar.to(device), joint_ar.to(device)


def resnet_predict_input_processing(current_data, joint_position, device):
    # current_data = robot_interface.get_camera_data()
    # rgb
    rgb_ar = current_data["rgb"][:, :, 0:3]
    rgb_ar = torch.tensor(rgb_ar)
    # depth
    depth_ar = current_data["depth"]
    depth_threshold = 7.0
    depth_ar[depth_ar > depth_threshold] = depth_threshold
    depth_ar = depth_ar.reshape(depth_ar.shape[0], depth_ar.shape[1], 1)
    depth_ar = torch.tensor(depth_ar)
    # joint
    joint_ar = joint_position
    joint_ar = torch.tensor(joint_ar)
    return rgb_ar.to(device), depth_ar.to(device), joint_ar.to(device)


def closest_grasp(grasps, ee_pos):
    ee_pos = ee_pos.A
    matrices = grasps - ee_pos
    dists = np.linalg.norm(matrices, axis=(1, 2))
    min_index = np.argmin(dists)
    return grasps[min_index]

def initial_object_pos_selector(mode="single"):
    # if area == 0 then the initial pose of object is random,
    # if area ==2 then the initial pose of object is fixed
    # area = random.randint(0, 0)
    if mode == "multiple":
        area = 3
    else:
        area = 0

    if area == 0:
        x = np.random.uniform(low=1.55, high=1.85, size=1)
        y = np.random.uniform(low=0.0, high=0.5, size=1)
        z = np.random.uniform(low=0.85, high=0.85, size=1)
        object_pose = [x[0], y[0], z[0]]
    elif area == 1:
        x = np.random.uniform(low=0, high=0.4, size=1)
        y = np.random.uniform(low=1.25, high=1.55, size=1)
        z = np.random.uniform(low=0, high=0, size=1)
        object_pose = [x[0], y[0], z[0]]
    elif area == 2:
        x = np.random.uniform(low=1.5, high=1.5, size=1)
        y = np.random.uniform(low=0.2, high=0.2, size=1)
        z = np.random.uniform(low=0.6, high=0.6, size=1)
        object_pose = [x[0], y[0], z[0]]
    elif area == 3:
        x = np.random.uniform(low=1.65, high=1.71, size=1)
        y = np.random.uniform(low=-0.10, high=0.40, size=1)
        z = np.random.uniform(low=0.80, high=0.80, size=1)
        object_pose = [x[0], y[0], z[0]]
    else:
        x = np.random.uniform(low=-1.4, high=-1.05, size=1)
        y = np.random.uniform(low=-0.05, high=0.45, size=1)
        z = np.random.uniform(low=0, high=0, size=1)
        object_pose = [x[0], y[0], z[0]]

    return object_pose


def wTgrasp_finder(suc_grasps, robot, new_object_pos):
    selected_grasp = closest_grasp(grasps=suc_grasps, ee_pos=robot.fkine(robot.q))
    wTobj = sm.SE3.Rt(
        sm.SO3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        new_object_pos,
    )

    objTgrasp = sm.SE3(trnorm(selected_grasp))
    graspTgoal = np.eye(4)
    fT = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    fT = sm.SE3(fT)
    graspTgoal[2, 3] = 0.1634
    # graspTgoal[0, 3] = -0.075

    graspTgoal = sm.SE3(graspTgoal)
    wTgrasp = wTobj * objTgrasp * graspTgoal * fT

    return wTgrasp


def save_rgb_depth(current_data, step_counter, success_counter):
    rgb_image = cv2.cvtColor(current_data["rgb"][:, :, 0:3], cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        "collected_data/traj{}/rgb/{}.png".format(success_counter, step_counter),
        rgb_image,
    )
    cv2.imwrite(
        "collected_data/traj{}/depth/{}.png".format(success_counter, step_counter),
        current_data["depth"],
    )


def save_rgb_depth_feedback(current_data, step_counter, success_counter):
    rgb_image = cv2.cvtColor(current_data["rgb"][:, :, 0:3], cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        "collected_data_feedback/traj{}/rgb/{}.png".format(
            success_counter, step_counter
        ),
        rgb_image,
    )
    cv2.imwrite(
        "collected_data_feedback/traj{}/depth/{}.png".format(
            success_counter, step_counter
        ),
        current_data["depth"],
    )


def update_pose_dict(pose_dict_name, step_counter, joint_pos, ee_twist):
    pose_dict_name[step_counter] = {}
    pose_dict_name[step_counter]["ee_twist"] = ee_twist
    # pose_dict_name[step_counter]["ee_pose"] = ee_pose
    pose_dict_name[step_counter]["joint_pos"] = joint_pos
    return pose_dict_name


def create_traj_folders_dict(success_counter):
    traj_path = os.path.join(
        os.getcwd() + "/collected_data",
        "traj{}".format(success_counter),
    )
    os.makedirs(traj_path, exist_ok=True)
    rgb_path = os.path.join(
        os.getcwd() + "/collected_data",
        "traj{}/rgb".format(success_counter),
    )
    os.makedirs(rgb_path, exist_ok=True)
    depth_path = os.path.join(
        os.getcwd() + "/collected_data",
        "traj{}/depth".format(success_counter),
    )
    os.makedirs(depth_path, exist_ok=True)
    pose_dict_name = "pose{}".format(success_counter)
    pose_dict_name = {}
    return traj_path, pose_dict_name


def update_pose_feedback_dict(
    pose_dict_name, feedback_dict_name, step_counter, joint_pos, ee_twist, gt_action
):
    pose_dict_name[step_counter] = {}
    pose_dict_name[step_counter]["ee_twist"] = ee_twist
    pose_dict_name[step_counter]["joint_pos"] = joint_pos
    feedback_dict_name[step_counter] = gt_action
    return pose_dict_name, feedback_dict_name


def dir_name_creator_feedback(traj_counter):
    traj_path = os.path.join(
        os.getcwd() + "/collected_data_feedback",
        "traj{}".format(traj_counter),
    )
    os.makedirs(traj_path, exist_ok=True)
    rgb_path = os.path.join(
        os.getcwd() + "/collected_data_feedback",
        "traj{}/rgb".format(traj_counter),
    )
    os.makedirs(rgb_path, exist_ok=True)
    depth_path = os.path.join(
        os.getcwd() + "/collected_data_feedback",
        "traj{}/depth".format(traj_counter),
    )
    os.makedirs(depth_path, exist_ok=True)

    pose_dict_name = "pose{}".format(traj_counter)
    pose_dict_name = {}

    feedback_dict_name = "feedback{}".format(traj_counter)
    feedback_dict_name = {}
    return traj_path, pose_dict_name, feedback_dict_name


def euler2mat(theta):
    """
    input
        theta = [theta1, theta2, theta3] = rotation angles in radian
    output
        matrix = 3x3 rotation matrix (numpy array)
    """
    theta1, theta2, theta3 = theta
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta1), -np.sin(theta1)],
            [0, np.sin(theta1), np.cos(theta1)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta2), 0, np.sin(theta2)],
            [0, 1, 0],
            [-np.sin(theta2), 0, np.cos(theta2)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta3), -np.sin(theta3), 0],
            [np.sin(theta3), np.cos(theta3), 0],
            [0, 0, 1],
        ]
    )
    matrix = np.dot(R_z, np.dot(R_y, R_x))
    return matrix


def mat2euler(matrix):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
    output
        theta1, theta2, theta3 = rotation angles in rotation order (xyz)
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    theta1 = np.arctan(-r23 / r33)
    theta2 = np.arctan(r13 * np.cos(theta1) / r33)
    theta3 = np.arctan(-r12 / r11)

    return np.array((theta1, theta2, theta3))


def get_usd_and_h5_paths(data_dir):
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    ]
    for file in files:
        if file.endswith(".usd"):
            usd_file = file
        elif file.endswith(".h5"):
            h5_file = file
    return usd_file, h5_file


def move_to_goal(interface, controller, goal_pose):
    """Move robot to goal pose."""
    ee_twist, error_t, error_rpy = controller.wTeegoal_2_eetwist(goal_pose)
    qd = controller.ee_twist_2_qd(ee_twist)
    interface.move_joints(qd)
    distance_lin = np.linalg.norm(error_t)
    distance_ang = np.linalg.norm(error_rpy)
    goal_reached = distance_lin < 0.02 and distance_ang < 0.01
    
    return distance_lin, goal_reached, ee_twist

def grasp_obj(robot_interface, controller):
    robot_interface.close_gripper()
    controller.grasp_counter += 1
    obj_grasped = True if controller.grasp_counter > 40 else False
    return obj_grasped

def create_traj_folders_dict_various_obj(obj_name, success_counter):
    traj_path = os.path.join(
        os.getcwd() + "/collected_data2",
        "{}{}".format(obj_name, success_counter),
    )
    os.makedirs(traj_path, exist_ok=True)
    rgb_path = os.path.join(
        os.getcwd() + "/collected_data2",
        "{}{}/rgb".format(obj_name, success_counter),
    )
    os.makedirs(rgb_path, exist_ok=True)
    depth_path = os.path.join(
        os.getcwd() + "/collected_data2",
        "{}{}/depth".format(obj_name, success_counter),
    )
    os.makedirs(depth_path, exist_ok=True)
    pose_dict_name = "pose{}".format(success_counter)
    pose_dict_name = {}
    return traj_path, pose_dict_name

def save_rgb_depth_various_obj(current_data, step_counter, success_counter, obj_name):
    rgb_image = cv2.cvtColor(current_data["rgb"][:, :, 0:3], cv2.COLOR_BGR2RGB)
    cv2.imwrite("collected_data2/{}{}/rgb/{}.png".format(obj_name, success_counter,
            step_counter), rgb_image)
    cv2.imwrite("collected_data2/{}{}/depth/{}.png".format(obj_name, success_counter,
            step_counter), current_data["depth"])

def set_seed(seed_value=42):
    # Set the seed for Python's built-in random module
    random.seed(seed_value)
    # Set the seed for NumPy
    np.random.seed(seed_value)
    # Set the seed for PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def object_selector(objects):
    # Select an object randomly
    object_index = np.random.randint(0, len(objects))
    selected_object = objects[object_index]
    return selected_object, object_index