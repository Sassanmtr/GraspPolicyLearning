import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import spatialmath as sm
from spatialmath.base import trnorm
import h5py
from scipy.spatial.transform import Rotation
from PIL import Image
import torch
from torchvision import transforms
import os
import fnmatch
from spatialmath.base import q2r
from spatialmath import SE3, SO3

def initial_object_pos_selector():
    area = random.randint(2, 2)    
    if area == 0:
        x = np.random.uniform(low=1.5, high=1.9, size=1)
        y = np.random.uniform(low=0.0, high=0.5, size=1)
        z = np.random.uniform(low=0.6, high=0.6, size=1)
        object_pose = [x[0], y[0], z[0]]
    elif area == 1:
        x = np.random.uniform(low=0, high=0.4, size=1)
        y = np.random.uniform(low=1.25, high=1.55, size=1)
        z = np.random.uniform(low=0, high=0, size=1)
        object_pose = [x[0], y[0], z[0]]
    elif area == 2:
        x = np.random.uniform(low=1.7, high=1.7, size=1)
        y = np.random.uniform(low=0.2, high=0.2, size=1)
        z = np.random.uniform(low=0.6, high=0.6, size=1)
        object_pose = [x[0], y[0], z[0]]
    else:
        x = np.random.uniform(low=-1.4, high=-1.05, size=1)
        y = np.random.uniform(low=-0.05, high=0.45, size=1)
        z = np.random.uniform(low=0, high=0, size=1)
        object_pose = [x[0], y[0], z[0]]
    return object_pose


def gripper_inital_point_selector():
    area = 2  #comment this line for random initial pose of gripper
    if area == 0:
        lower = np.array([1.05, 0.21, 1.19])
        upper = np.array([0.95, 0.18, 1.17])
        theta = -50
        r = R.from_euler("xyz", [180, theta, 0], degrees=True)
    elif area == 1:
        lower = np.array([0.2, 0.7, 0.9])
        upper = np.array([0.2, 0.7, 0.9])
        theta = 40
        r = R.from_euler("xyz", [180, theta, 90], degrees=True)
    elif area == 2:
        lower = np.array([1.0, 0.20, 1.18])
        upper = np.array([1.0, 0.20, 1.18])
        theta = -50
        r = R.from_euler("xyz", [180, theta, 0], degrees=True)
    else:
        lower = np.array([-0.5, 0.2, 0.9])
        upper = np.array([-0.5, 0.2, 0.9])
        theta = -40
        r = R.from_euler("xyz", [180, theta, 180], degrees=True)
    x = np.random.uniform(low=lower[0], high=upper[0], size=1)
    y = np.random.uniform(low=lower[1], high=upper[1], size=1)
    z = np.random.uniform(low=lower[2], high=upper[2], size=1)
    target_ori = sm.SO3(np.array(r.as_matrix()))
    target_pos = [x[0], y[0], z[0]]
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose


def get_observations(robot, obj):
    obj_position, obj_orientation = obj.get_world_pose()
    end_effector_pose = robot.fkine(robot.q)
    vertical_grasp = sm.SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    observations = {
        robot.name: {
            "end_effector_position": end_effector_pose,
        },
        obj.name: {
            "position": obj_position,
            "orientation": obj_orientation,
            "target_position": sm.SE3.Rt(vertical_grasp, [0.6, -1.5, 0.7]),
        },
    }
    return observations


def wTgrasp_finder(suc_grasps, robot, new_object_pos):
    selected_grasp = closest_grasp(
        grasps=suc_grasps,
        ee_pos=robot.fkine(robot.q),
    )
    wTobj = sm.SE3.Rt(
        sm.SO3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        new_object_pos,
    )

    objTgrasp = sm.SE3(trnorm(selected_grasp))
    graspTgoal = np.eye(4)
    fT = np.array(
        [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    fT = sm.SE3(fT)
    graspTgoal[2, 3] = 0.1634

    graspTgoal = sm.SE3(graspTgoal)
    wTgrasp = wTobj * objTgrasp * graspTgoal * fT
    
    return wTgrasp


def closest_grasp(grasps, ee_pos):
    ee_pos = ee_pos.A
    matrices = grasps - ee_pos
    dists = np.linalg.norm(matrices, axis=(1, 2))
    min_index = np.argmin(dists)
    return grasps[min_index]


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


def experience_collector(traj_dir):
    rgb_dir = os.path.join(traj_dir, "rgb")
    image_data = []
    joint_data = []
    ee_pose_data = []
    gripper_pos = []
    action = []
    # number of files in a folder
    for i in range(202):
        image_step, joint_step, ee_pose_step = data_collector(rgb_dir, i)
        image_data.append(image_step)
        joint_data.append(torch.tensor(joint_step))
        ee_pose_data.append(ee_pose_step)
        gripper_pos.append(joint_step[-2] + joint_step[-1])
    image_data.pop()
    joint_data.pop()
    gripper_pos = gripper_pos[1:]
    ee_pose_data = np.array(ee_pose_data)

    old_action = ee_pose_data[1:] - ee_pose_data[:-1]
    for j in range(len(gripper_pos)):
        if gripper_pos[j] > 0.04:
            action.append(np.concatenate((old_action[j], np.array([0.8]).reshape(1,1)), axis=1))
        else:
            action.append(np.concatenate((old_action[j], np.array([-0.8]).reshape(1,1)), axis=1))
    
    # for i in range(ee_pose_data.shape[0] - 1):
    #     diff_pos = ee_pose_data[i + 1,:,:3] - ee_pose_data[i,:,:3]
    #     # Computing the relative quaternion
    #     diff_rel = Rotation.from_quat(ee_pose_data[i,:,3:]).inv() * Rotation.from_quat(ee_pose_data[i+1,:,3:])
    #     old_action = np.concatenate((diff_pos, diff_rel.as_quat()), axis=1)
    #     if gripper_pos[i] > 0.04:
    #         action.append(np.concatenate((old_action, np.array([0.8]).reshape(1,1)), axis=1))
    #     else:
    #         action.append(np.concatenate((old_action, np.array([-0.8]).reshape(1,1)), axis=1))
    
    return torch.stack(image_data), torch.stack(joint_data), torch.tensor(action)

def data_collector(rgb_dir, step_index):
    # image (rgb and depth)
    convert_tensor = transforms.ToTensor()
    rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
    rgb_img = convert_tensor(rgb_img)
    traj_dir = rgb_dir[:-4]
    depth_dir = os.path.join(traj_dir, "depth")
    depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
    depth_img = convert_tensor(depth_img)           
    im_info = torch.cat((rgb_img, depth_img), dim=0)
    #joint
    pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
    joint_info = pose_file.item()[step_index]["joint_pos"]
    #action
    current_ee_pose_SM = pose_file.item()[step_index]["ee_pose"]
    t_ee_pose = current_ee_pose_SM.t.reshape(1, 3)
    R_ee_pose = current_ee_pose_SM.R
    R_ee_pose = SO3(R_ee_pose).eul().reshape(1,3)
    # for Euler start removing here
    # r_ee_matrix = Rotation.from_matrix(R_ee_pose)
    # r_ee_quat = r_ee_matrix.as_quat()
    # finish removing here and uncomment the below line and change the second element of current_ee_pose
    # R_ee_pose = rotation_angles(R_ee_pose).reshape(1, 3)
    # current_ee_pose = np.concatenate((t_ee_pose, r_ee_quat.reshape(1, 4)), axis=-1)
    current_ee_pose = np.concatenate((t_ee_pose, R_ee_pose), axis=-1)

    return im_info, joint_info, current_ee_pose

def index_sampler(traj_dir, sequence_len):
    traj_len = len(fnmatch.filter(os.listdir(traj_dir), '*.*'))
    if traj_len <= sequence_len:
        raise "trajectory length must be greater than the sequence length" 
    last_valid_index = int(traj_len - sequence_len - 1)
    index = random.randint(0, last_valid_index)
    return index


def q_output_processing(current_ee_pose, next_action):
    current_ori = current_ee_pose.R
    current_pos = current_ee_pose.t
    diff_ori = q2r(np.array(next_action[0,3:-1]))
    target_ori = sm.SO3(trnorm(diff_ori * current_ori))
    target_pos = [next_action[0,0] + current_pos[0], next_action[0,1] + current_pos[1], next_action[0,2] + current_pos[2]] 
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose

def output_processing(current_ee_pose, next_action):
    rot = current_ee_pose.R
    current_ori = R.from_matrix(rot)
    current_ori = current_ori.as_euler("xyz")
    current_pos = current_ee_pose.t
    diff_ori = np.array([next_action[0, 3], next_action[0, 4], next_action[0, 5]])
    target_ori = [current_ori[0] + diff_ori[0], current_ori[1] + diff_ori[1], current_ori[2] + diff_ori[2]]
    target_pos = [next_action[0, 0] + current_pos[0], next_action[0, 1] + current_pos[1], next_action[0, 2] + current_pos[2]] 
    r = R.from_euler("xyz", [target_ori[0], target_ori[1], target_ori[2]])
    target_ori = sm.SO3(trnorm(np.array(r.as_matrix())))
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose

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

    return np.array((theta1, theta2, theta3))
