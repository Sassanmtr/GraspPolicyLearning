import os
import numpy as np
import torch
from PIL import Image
import cv2
import h5py
from scipy.spatial.transform import Rotation as R
import spatialmath as sm
import wandb
from collections import deque
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
    lower = np.array([1.05, 0.21, 1.19])
    upper = np.array([0.95, 0.18, 1.17])
    theta = -50
    r = R.from_euler("xyz", [180, theta, 0], degrees=True)
    x = np.random.uniform(low=lower[0], high=upper[0], size=1)
    y = np.random.uniform(low=lower[1], high=upper[1], size=1)
    z = np.random.uniform(low=lower[2], high=upper[2], size=1)
    target_ori = sm.SO3(np.array(r.as_matrix()))
    target_pos = [x[0], y[0], z[0]]
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose

def output_processing(current_ee_pose, next_action):
    rot = current_ee_pose.R
    current_ori = R.from_matrix(rot)
    current_ori = current_ori.as_euler("xyz")
    current_pos = current_ee_pose.t
    diff_ori = np.array([next_action[3], next_action[4], next_action[5]])
    target_ori = [current_ori[0] + diff_ori[0], current_ori[1] + diff_ori[1], current_ori[2] + diff_ori[2]]
    target_pos = [next_action[0] + current_pos[0], next_action[1] + current_pos[1], next_action[2] + current_pos[2]] 
    r = R.from_euler("xyz", [target_ori[0], target_ori[1], target_ori[2]])
    target_ori = sm.SO3(trnorm(np.array(r.as_matrix())))
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
    print("Depth max: ", torch.max(depth_ar))
    im_ar = im_ar.permute(2, 0, 1)
    joint_ar = robot_sim.get_joint_positions()
    joint_ar = torch.tensor(joint_ar)
    return im_ar.to(device), joint_ar.to(device)

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

def save_image_update_pose(robot, robot_interface, robot_sim, pose_dict_name, step_counter, traj_counter):    
    current_data = robot_interface.get_camera_data()
    rgb_im = Image.fromarray(current_data["rgb"][:, :, 0:3])
    rgb_im = (rgb_im * 256.0).astype(np.uint16)
    rgb_im.save(
        "collected_data/traj{}/rgb/{}.jpeg".format(
            traj_counter, step_counter
        )
    )
    depth_im = current_data["depth"]
    depth_threshold = 7.0
    depth_im[depth_im > depth_threshold] = depth_threshold
    depth_im = (depth_im * 100.0).astype(np.uint16)
    cv2.imwrite(
        "collected_data/traj{}/depth/{}.jpeg".format(
            traj_counter, step_counter
        ),
        depth_im,
    )
    pose_dict_name[step_counter] = {}
    pose_dict_name[step_counter]["ee_pose"] = robot.fkine(robot.q)
    pose_dict_name[step_counter][
        "joint_pos"
    ] = robot_sim.get_joint_positions()


def closest_grasp(grasps, ee_pos):
    ee_pos = ee_pos.A
    matrices = grasps - ee_pos
    dists = np.linalg.norm(matrices, axis=(1, 2))
    min_index = np.argmin(dists)
    return grasps[min_index]


def initial_object_pos_selector():
    x = np.random.uniform(low=1.5, high=1.9, size=1)
    y = np.random.uniform(low=0.0, high=0.5, size=1)
    z = np.random.uniform(low=0.6, high=0.6, size=1)
    object_pose = [x[0], y[0], z[0]]
    return object_pose

def dir_name_creator(traj_counter):
    traj_path = os.path.join(
        os.getcwd() + "/collected_data",
        "traj{}".format(traj_counter),
    )
    os.mkdir(traj_path)
    rgb_path = os.path.join(
        os.getcwd() + "/collected_data",
        "traj{}/rgb".format(traj_counter),
    )
    os.mkdir(rgb_path)
    depth_path = os.path.join(
        os.getcwd() + "/collected_data",
        "traj{}/depth".format(traj_counter),
    )
    os.mkdir(depth_path)

    pose_dict_name = "pose{}".format(traj_counter)
    pose_dict_name = {}

    feedback_dict_name = "feedback{}".format(traj_counter)
    feedback_dict_name = {}
    return traj_path, pose_dict_name, feedback_dict_name

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


def save_pose_feedback(pose_dict_name, feedback_dict_name, traj_counter):

    for j in range(len(feedback_dict_name.keys())):
        feedback_info = feedback_dict_name[j]
        if np.isnan(feedback_info).any():
            print("FEEDBACK")
            print("step: ", j)
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOO!")

    np.save(
        "collected_data/traj{}/pose.npy".format(traj_counter),
        pose_dict_name,
    )
    np.save("collected_data/traj{}/feedback.npy".format(traj_counter), feedback_dict_name)
    print("traj_counter: ", traj_counter)