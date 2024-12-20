import argparse
import sys
import os
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
import yaml
from yaml.loader import SafeLoader
import torch
import wandb
import threading
from utils.helpers import *
from utils.isaac_loader import *
from bc_network.networks import PolicyTwist
from bc_network.datasets import ReplayBufferTwistFeedback

set_seed(42)

def train_step(policy, replay_memory, config):
    epoch_counter = 0
    save_interval = 1000
    while len(replay_memory) < 1200:
        batch = replay_memory.sample(config["batch_size"])
        rgb_batch, depth_batch, proprio_batch, action_batch = batch
        training_metrics = policy.update_params(
            rgb_batch, depth_batch, proprio_batch, action_batch
        )
        wandb.log({"training loss": training_metrics})
        epoch_counter += 1
        if epoch_counter % save_interval == 0:
            file_name = "saved_models/" + "policy_feedback_multiple.pt"
            torch.save(policy.state_dict(), file_name)
            print(f"Saved model at epoch {epoch_counter}")

def simulation_main(policy, replay_memory, lstm_state1=None, lstm_state2=None):
    simulation_app = SimulationApp({"headless":True})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    import omni
    from utils.fmm_isaac import FmmIsaacInterface
    from FmmControlLite.fmm_control_lite.fmm_control import FmmQPControl

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit()

    # Initialize Isaac 
    my_world = initialize_world(World)
    my_robot = initialize_robot(my_world, Robot, add_reference_to_stage)
    initialize_environment(my_world, add_reference_to_stage, XFormPrim)
    objects, suc_grasps = initialize_multiple_objects_grasps(my_world, add_reference_to_stage, RigidPrim, XFormPrim, omni, utils)

    # Initialize Controller
    my_world.reset()
    robot_interface = FmmIsaacInterface(my_robot)
    lite_controller = FmmQPControl(
        dt=(1.0 / 40.0), fmm_mode="no-tower", robot=robot_interface.robot_model
    )

    start_simulation(my_world)
    nr_traj = 1000
    traj_counter = len(replay_memory)
    step_threshold = 500
    success_counter = 0
    failure_counter = 0
    grasp_counter = 0
    nav_flag = True
    pregrasp_flag = False
    grasp_flag = False
    pick_flag = False
    success_flag = False
    successful_ind = []
    for _ in range(2):
        gt = robot_interface.get_camera_data()
        print("gt: ", gt["rgb"].shape)
        my_world.step(render=True)
    while simulation_app.is_running():
        _, object_index = object_selector(objects)
        my_world.reset()
        my_world.step(render=True)
        new_object_pos = initial_object_pos_selector(mode="multiple")
        ee_initial_target_pose = gripper_inital_point_selector()
        objects[object_index].set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
        step_counter = 0
        if success_counter + failure_counter == nr_traj:
            print("Enough trajectories have been recorded!")
            break
            
        # Create directory to save data
        traj_path, pose_dict_name, feedback_dict_name = dir_name_creator_feedback(traj_counter)
        while True:
            if step_counter == step_threshold or success_flag:
                np.save("collected_data_feedback/traj{}/pose.npy".format(traj_counter), pose_dict_name)
                np.save("collected_data_feedback/traj{}/feedback.npy".format(traj_counter), feedback_dict_name)
                replay_memory.append(traj_path)
                print("Trajectory {} is recorded!".format(traj_counter))
                if success_flag:
                    success_counter += 1
                    successful_ind.append(traj_counter)
                    traj_counter += 1
                else:
                    failure_counter += 1
                    traj_counter += 1
                nav_flag = True
                success_flag = False
                pregrasp_flag = False
                grasp_flag = False
                pick_flag = False
                print("Success: ", success_counter, "Failure: ", failure_counter)
                print("Successful indices: ", successful_ind)
                break
            if nav_flag:
                init_dist, _, _ = move_to_goal(
                    robot_interface, lite_controller, ee_initial_target_pose
                )
                robot_interface.update_robot_model()
                my_world.step(render=True)
                if init_dist < 0.027:
                    nav_flag = False
                    pregrasp_flag = True
                    grasp_pose = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)
                    pick_position = [grasp_pose.t[0], grasp_pose.t[1], 1.2]
                    current_ori = grasp_pose.R
                    pick_pose = sm.SE3.Rt(current_ori, pick_position, check=False)
            
            if pregrasp_flag:
                gt = robot_interface.get_camera_data()
                joint_pos = my_robot.get_joint_positions()
                gt_action, grasp_dist, _ = lite_controller.wTeegoal_2_eetwist(grasp_pose)
                rgb_array, depth_array, joint_array = resnet_predict_input_processing(gt, joint_pos, device)
                next_action, lstm_state1, lstm_state2 = policy.predict(rgb_array, depth_array, joint_array, lstm_state1, lstm_state2)
                save_rgb_depth_feedback(gt, step_counter, traj_counter)
                pose_dict_name, feedback_dict_name = update_pose_feedback_dict(
                    pose_dict_name, feedback_dict_name, step_counter, joint_pos, next_action, gt_action)
                feedback_dict_name[step_counter] = np.append(feedback_dict_name[step_counter], [1.0]).reshape(1, -1)
                if np.linalg.norm(grasp_dist) < 0.03:
                    pregrasp_flag = False
                    grasp_flag = True
                qd = lite_controller.ee_twist_2_qd(next_action[:-1])
                robot_interface.move_joints(qd)
                if next_action[-1] == 0.0:
                    robot_interface.close_gripper()
                robot_interface.update_robot_model()
                my_world.step(render=True)
                step_counter += 1

            if grasp_flag:
                gt = robot_interface.get_camera_data()
                joint_pos = my_robot.get_joint_positions()
                gt_action, grasp_dist, _ = lite_controller.wTeegoal_2_eetwist(grasp_pose)
                rgb_array, depth_array, joint_array = resnet_predict_input_processing(gt, joint_pos, device)
                next_action, lstm_state1, lstm_state2 = policy.predict(rgb_array, depth_array, joint_array, lstm_state1, lstm_state2)
                save_rgb_depth_feedback(gt, step_counter, traj_counter)
                pose_dict_name, feedback_dict_name = update_pose_feedback_dict(
                    pose_dict_name, feedback_dict_name, step_counter, joint_pos, next_action, gt_action)
                feedback_dict_name[step_counter] = np.append(feedback_dict_name[step_counter], [0.0]).reshape(1, -1)
                grasp_counter += 1
                if grasp_counter == 40:
                    pick_flag = True
                    grasp_flag = False
                    grasp_counter = 0
                qd = lite_controller.ee_twist_2_qd(next_action[:-1])
                robot_interface.move_joints(qd)
                if next_action[-1] == 0.0:
                    robot_interface.close_gripper()
                robot_interface.update_robot_model()
                my_world.step(render=True)
                step_counter += 1
            if pick_flag:
                gt = robot_interface.get_camera_data()
                joint_pos = my_robot.get_joint_positions()
                gt_action, grasp_dist, _ = lite_controller.wTeegoal_2_eetwist(pick_pose)
                rgb_array, depth_array, joint_array = resnet_predict_input_processing(gt, joint_pos, device)
                next_action, lstm_state1, lstm_state2 = policy.predict(rgb_array, depth_array, joint_array, lstm_state1, lstm_state2)
                save_rgb_depth_feedback(gt, step_counter, traj_counter)
                pose_dict_name, feedback_dict_name = update_pose_feedback_dict(
                    pose_dict_name, feedback_dict_name, step_counter, joint_pos, next_action, gt_action)
                feedback_dict_name[step_counter] = np.append(feedback_dict_name[step_counter], [0.0]).reshape(1, -1)
                qd = lite_controller.ee_twist_2_qd(next_action[:-1])
                robot_interface.move_joints(qd)
                if next_action[-1] == 0.0:
                    # print("close gripper")
                    robot_interface.close_gripper()
                robot_interface.update_robot_model()
                my_world.step(render=True)
                step_counter += 1
                if objects[object_index].get_world_pose()[0][-1] > 0.95:
                    success_flag = True
                    pick_flag = False
    simulation_app.close()
    return

def load_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=SafeLoader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Behavior Cloning in Simulation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save the trained models")
    parser.add_argument("--use_wandb", action='store_true', help="Flag to use wandb for logging")
    args = parser.parse_args()

    config_path = Path.cwd() / "config.yaml"
    net_config = load_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    replay_memory = ReplayBufferTwistFeedback(net_config["buffer_capacity"], args.data_dir, net_config["sequence_len"])
    for subdir in os.listdir(args.data_dir):
        subdir_path = os.path.join(args.data_dir, subdir)
        if os.path.isdir(subdir_path):
            replay_memory.append(subdir_path)

    policy = PolicyTwist(net_config, device)
    policy.load_state_dict(torch.load(args.model_path))

    if args.use_wandb:
        wandb.init(project="Interactive Imiation Learning Multiple Objects")
        wandb.watch(policy, log_freq=100)

    training_thread = threading.Thread(target=train_step, args=(policy, replay_memory, net_config, args.save_dir))
    training_thread.start()

    simulation_main(policy, replay_memory, device)

    training_thread.join()
