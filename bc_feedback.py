import sys
import os
from pathlib import Path
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
import yaml
from yaml.loader import SafeLoader
import torch
import wandb
import threading
from utils.helpers import *
from bc_network.bcnet import Policy_twist
from bc_network.bcdataset import ReplayBuffer_twist_feedback
HOME = str(Path.home())
print("HOME: ", HOME)

def train_step(policy, replay_memory, config):
    epoch_counter = 0
    save_interval = 100
    while True:
        batch = replay_memory.sample(config["batch_size"])
        camera_batch, proprio_batch, action_batch = batch
        training_metrics = policy.update_params(
            camera_batch, proprio_batch, action_batch
        )
        # wandb.log({"training loss": training_metrics})
        epoch_counter += 1
        if epoch_counter % save_interval == 0:
            file_name = "saved_models/" + "policy_feedback.pt"
            torch.save(policy.state_dict(), file_name)
            print(f"Saved model at epoch {epoch_counter}")

def simulation_main(config, policy, replay_memory, lstm_state=None):
    simulation_app = SimulationApp({"headless":False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    import omni.usd
    from src.fmm_isaac import FmmIsaacInterface
    from fmm_control_lite.fmm_control import FmmQPControl

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit()

    # Initialize World
    world_settings = {
        "stage_units_in_meters": 1.0,
        "physics_dt": 1.0 / config["fps"],
        "rendering_dt": 1.0 / config["fps"],
    }
    my_world = World(**world_settings)
    my_world.scene.add_default_ground_plane()

    # Initialize Robot
    asset_path = config["model_path"]
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/FMM")
    robot_sim = my_world.scene.add(
        Robot(prim_path="/World/FMM", name="fmm", position=[0, 0, 0])
    )

    # Initialize Env
    add_reference_to_stage(
        usd_path=HOME + "/Documents/isaac-fmm/models/hospital.usd",
        prim_path="/World/Hospital",
    )
    my_world.scene.add(
        XFormPrim(prim_path="/World/Hospital", name="hospital", position=[0, 0, 0])
    )

    suc_grasps, object_position, object_scale, object_mass = mesh_data(
        config["mesh_dir"]
    )

    add_reference_to_stage(
        usd_path=config["object_path"], prim_path="/World/Hospital/object"
    )

    my_object = my_world.scene.add(
        RigidPrim(
            prim_path="/World/Hospital/object",
            name="fancy_bucket",
            position=(0, 0, 0),
            mass=object_mass * 100,
        )
    )

    my_object = my_world.scene.add(
        XFormPrim(
            prim_path="/World/Hospital/object",
            name="fancy_bowl",
            position=object_position + np.array([0, 0, 0]),
            scale=(object_scale, object_scale, object_scale),
            visible=True,
        )
    )

    print("Adding PHYSICS to ShapeNet model")
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim("/World/Hospital/object", "Xform")
    shape_approximation = "convexDecomposition"
    utils.setRigidBody(prim, shape_approximation, False)
    print("CHECK mesh: ", my_world.scene.object_exists(name="fancy_bowl"))

    # Initialize Controller
    my_world.reset()
    robot_interface = FmmIsaacInterface(robot_sim)
    lite_controller = FmmQPControl(
        dt=(1.0 / config["fps"]), fmm_mode="no-tower", robot=robot_interface.robot_model
    )

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    nr_traj = 200
    traj_counter = len(replay_memory)
    step_threshold = 200
    success_counter = 0
    failure_counter = 0
    grasp_counter = 0
    nav_flag = True
    grasp_flag = False
    pick_flag = False
    success_flag = False
    for i in range(2):
        gt = robot_interface.get_camera_data()
        print("gt: ", gt["rgb"].shape)
        my_world.step(render=True)
    while simulation_app.is_running():
        my_world.reset()
        my_world.step(render=True)
        new_object_pos = initial_object_pos_selector()
        ee_initial_target_pose = gripper_inital_point_selector()
        my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
        step_counter = 0
        if success_counter + failure_counter == nr_traj:
            print("Enough trajectories have been recorded!")
            break
            
        # Create directory to save data
        traj_path, pose_dict_name, feedback_dict_name = dir_name_creator_feedback(traj_counter)
        while True:
            if step_counter == step_threshold or success_flag:
                np.save("collected_data/traj{}/pose.npy".format(traj_counter), pose_dict_name)
                np.save("collected_data/traj{}/feedback.npy".format(traj_counter), feedback_dict_name)
                replay_memory.append(traj_path)
                grasp_flag = False
                print("Trajectory {} is recorded!".format(traj_counter))
                if success_flag:
                    success_counter += 1
                    traj_counter += 1
                else:
                    failure_counter += 1
                    traj_counter += 1
                nav_flag = True
                break
            if nav_flag:
                init_dist, goal_reached, ee_twist = move_to_goal(
                    robot_interface, lite_controller, ee_initial_target_pose
                )
                robot_interface.update_robot_model()
                my_world.step(render=True)
                if init_dist < 0.027:
                    nav_flag = False
                    grasp_flag = True
                    grasp_pose = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)
                    pick_position = [grasp_pose.t[0], grasp_pose.t[1], 1.2]
                    current_ori = grasp_pose.R
                    pick_pose = sm.SE3.Rt(current_ori, pick_position, check=False)
            
            if grasp_flag:
                gt = robot_interface.get_camera_data()
                image_array, joint_array = predict_input_processing(robot_interface, robot_sim, device)
                # print("my_world.current_time_step_index", my_world.current_time_step_index)
                gt_action, grasp_dist, _ = lite_controller.wTeegoal_2_eetwist(grasp_pose)
                next_action, lstm_state = policy.predict(image_array, joint_array, lstm_state)
                qd = lite_controller.ee_twist_2_qd(next_action[:-1])
                robot_interface.move_joints(qd)
                robot_interface.update_robot_model()
                my_world.step(render=True)
                save_rgb_depth(gt, step_counter, traj_counter)
                pose_dict_name, feedback_dict_name = update_pose_feedback_dict(
                    pose_dict_name, feedback_dict_name, step_counter, robot_sim, next_action, gt_action)
                if np.linalg.norm(grasp_dist) < 0.03:
                    feedback_dict_name[step_counter] = np.append(feedback_dict_name[step_counter], [-1]).reshape(1, -1)
                    grasp_counter += 1
                    if grasp_counter == 50:
                        pick_flag = True
                        grasp_flag = False
                        grasp_counter = 0

                else:
                    feedback_dict_name[step_counter] = np.append(feedback_dict_name[step_counter], [1]).reshape(1, -1)
                    
                step_counter += 1
                # print("step_counter", step_counter)
            if pick_flag:
                gt = robot_interface.get_camera_data()
                image_array, joint_array = predict_input_processing(robot_interface, robot_sim, device)
                # print("my_world.current_time_step_index", my_world.current_time_step_index)
                gt_action, grasp_dist, _ = lite_controller.wTeegoal_2_eetwist(pick_pose)
                next_action, lstm_state = policy.predict(image_array, joint_array, lstm_state)
                qd = lite_controller.ee_twist_2_qd(next_action[:-1])
                robot_interface.move_joints(qd)
                robot_interface.update_robot_model()
                my_world.step(render=True)
                save_rgb_depth(gt, step_counter, traj_counter)
                pose_dict_name, feedback_dict_name = update_pose_feedback_dict(
                    pose_dict_name, feedback_dict_name, step_counter, robot_sim, next_action, gt_action)
                feedback_dict_name[step_counter] = np.append(feedback_dict_name[step_counter], [-1]).reshape(1, -1)
                step_counter += 1
                # print("step_counter", step_counter)
                if my_object.get_world_pose()[0][-1] > 0.98:
                    success_flag = True
                    pick_flag = False
    simulation_app.close()
    return

def move_to_goal(interface, controller, goal_pose):
    """Move robot to goal pose."""
    ee_twist, error_t, error_rpy = controller.wTeegoal_2_eetwist(goal_pose)
    qd = controller.ee_twist_2_qd(ee_twist)
    interface.move_joints(qd)
    distance_lin = np.linalg.norm(error_t)
    distance_ang = np.linalg.norm(error_rpy)
    goal_reached = distance_lin < 0.02 and distance_ang < 0.01
    
    return distance_lin, goal_reached, ee_twist


if __name__ == "__main__":
    with open('simulation_config.yaml') as f:
        simulation_config = yaml.load(f, Loader=SafeLoader)
        print("simulation config: ", simulation_config)
    with open('config.yaml') as f:
        network_config = yaml.load(f, Loader=SafeLoader)
        print("network config: ", network_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "/home/mokhtars/Documents/bc_network/collected_data"
    replay_memory = ReplayBuffer_twist_feedback(network_config["buffer_capacity"], data_dir, network_config["sequence_len"])
    # Add trajectories to the replay_memory
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            replay_memory.append(subdir_path)
        
    policy = Policy_twist(network_config, device)
    model_path = "saved_models/policy.pt"
    policy.load_state_dict(torch.load(model_path))
    # wandb.init(project="Feedback BC")
    # wandb.watch(policy, log_freq=100)

    # training_loop = threading.Thread(
    #     target=train_step,
    #     args=(policy, replay_memory, network_config),
    # )
    # training_loop.start()

    simulation_main(simulation_config, policy, replay_memory)
    