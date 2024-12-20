import argparse
import yaml
import torch
from yaml.loader import SafeLoader
import numpy as np
from omni.isaac.kit import SimulationApp
from utils.helpers import *
from utils.isaac_loader import *
from bc_network.networks import PolicyTwist
import plotly.graph_objs as go
import wandb

set_seed(42)

def main(policy, lstm_state=None, lstm_state2=None, log=True):
    simulation_app = SimulationApp({"headless": True})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.physx.scripts import utils  # type: ignore
    import omni
    from utils.fmm_isaac import FmmIsaacInterface
    from FmmControlLite.fmm_control_lite.fmm_control import FmmQPControl

    # Initialize Isaac 
    my_world = initialize_world(World)
    my_robot = initialize_robot(my_world, Robot, add_reference_to_stage)
    initialize_environment(my_world, add_reference_to_stage, XFormPrim)
    my_object, suc_grasps = initialize_object_grasps(my_world, add_reference_to_stage, RigidPrim, XFormPrim, omni, utils) 

    # Initialize Controller
    my_world.reset()
    robot_interface = FmmIsaacInterface(my_robot)
    lite_controller = FmmQPControl(
        dt=(1.0 / 40.0), fmm_mode="no-tower", robot=robot_interface.robot_model
    )

    # Start simulation
    start_simulation(my_world)
    
    nr_traj = 100
    step_threshold = 500
    step_counter = 0
    traj_counter = 0
    success_counter = 0
    failure_counter = 0
    success_flag = False
    target_height = 0.96
    for _ in range(2):
        robot_interface.get_camera_data()
        my_world.step(render=True)
    results = []
    grasp_progress_ratio = 0
    pick_progress_ratio = 0
    final_grasp_dist = 100
    final_pick_dist = 100
    if log:
        wandb.init(project="Grasping Validation", name="twist single")
    while simulation_app.is_running():
        while traj_counter < nr_traj:
            my_world.reset()
            robot_interface.update_robot_model()
            my_world.step(render=True)
            new_object_pos = initial_object_pos_selector()
            init_pose = gripper_inital_point_selector()
            my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
            init_dist = 1
            
            if success_flag or step_counter == step_threshold:
                if success_flag:
                    success_counter += 1
                    res = 1
                else:
                    failure_counter += 1
                    res = 0
                results.append(res)
                grasp_progress_ratio = (initial_grasp_dist - final_grasp_dist) / initial_grasp_dist
                pick_progress_ratio = (initial_pick_dist - final_pick_dist) / initial_pick_dist
                traj_counter += 1
                if log:
                    wandb.log({"Progress Ratio Towards Grasp Pose": grasp_progress_ratio})
                    wandb.log({"Progress Ratio Towards Pick Pose": pick_progress_ratio})
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=["Success"], y=[success_counter], name='Success'))
                    fig.add_trace(go.Bar(x=["Failure"], y=[failure_counter], name='Failure'))
                    # Create a scatter plot for the success ratio
                    success_ratio = success_counter / traj_counter
                    fig.add_trace(go.Scatter(x=["Success", "Failure"], y=[success_ratio, success_ratio], mode='lines', name='Success Ratio', yaxis='y2', line=dict(width=6)))
                    # Update the layout to include the secondary y-axis for the success ratio
                    fig.update_layout(
                        yaxis=dict(title='Number of Trajectories'),
                        yaxis2=dict(title='Success Ratio', overlaying='y', side='right', range=[0, 1])
                    )
                    wandb.log({'Trajectory Chart': fig})
                success_flag = False
                step_counter = 0
                final_grasp_dist = 100
                final_pick_dist = 100
                continue
            
            while init_dist > 0.03:
                init_dist, _, _ = move_to_goal(
                    robot_interface, lite_controller, init_pose
                )
                robot_interface.update_robot_model()
                my_world.step(render=True)
            print("Reached initial goal")
            grasp_pose = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)
            initial_grasp_dist = np.linalg.norm(robot_interface.robot_model.fkine(robot_interface.robot_model.q).t - grasp_pose.t)
            initial_pick_dist = target_height - my_object.get_world_pose()[0][-1]
            while step_counter < step_threshold:
                current_data = robot_interface.get_camera_data()
                joint_pos = my_robot.get_joint_positions()
                rgb_array, depth_array, joint_array = resnet_predict_input_processing(current_data, joint_pos, device)
                next_action, lstm_state, lstm_state2 = policy.predict(
                    rgb_array, depth_array, joint_array, lstm_state, lstm_state2)
                qd = lite_controller.ee_twist_2_qd(next_action[:-1])
                robot_interface.move_joints(qd)
                if next_action[-1] == 0.0:
                    robot_interface.close_gripper()
                robot_interface.update_robot_model()
                my_world.step(render=True)
                step_counter += 1
                pick_dist = np.abs(target_height - my_object.get_world_pose()[0][-1])
                grasp_dist = np.linalg.norm(robot_interface.robot_model.fkine(robot_interface.robot_model.q).t - grasp_pose.t)
                if grasp_dist < final_grasp_dist:
                    final_grasp_dist = grasp_dist
                if pick_dist < final_pick_dist:
                    final_pick_dist = pick_dist
                print("final_dist: ", final_grasp_dist)

                if my_object.get_world_pose()[0][-1] > target_height:
                    success_flag = True
                    print("Success in ", success_counter)
                    break

        accuracy_ratio = success_counter / nr_traj
        if log:
            wandb.log({"Accuracy Ratio": accuracy_ratio})  
        # Exit simulation
        break
    if log:
        wandb.finish()
    simulation_app.close()
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a policy model for robotic grasping")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--log", action="store_true", help="Use wandb for logging")
    args = parser.parse_args()

    config_path = Path.cwd() / "config.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
        print("Config:", config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyTwist(config, device)
    policy.load_state_dict(torch.load(args.model_path))
    main(policy, args.log)
