import numpy as np
import spatialmath as sm
import shutil
from omni.isaac.kit import SimulationApp
from utils.helpers import *
from utils.isaac_loader import *


def main(nr_traj, step_threshold):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.physx.scripts import utils 
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

    start_simulation(my_world)
    success_counter = 0
    failure_counter = 0
    failure_flag = False
    obj_grasped = False
    for _ in range(2):
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
        if success_counter == nr_traj:
            print("SUCCESS, enough successful grasps")
            break
        # Create directory to save data
        traj_path, pose_dict_name = create_traj_folders_dict(success_counter)
        nav_flag = True
        pregrasp_flag = False
        grasp_flag = False
        pick_flag = False
        while step_counter <= step_threshold:
            if failure_flag or step_counter == step_threshold:
                shutil.rmtree(traj_path)
                failure_counter += 1
                print("Failure ", failure_counter)
                failure_flag = False
                break
            if nav_flag:
                init_dist, _, ee_twist = move_to_goal(
                    robot_interface, lite_controller, ee_initial_target_pose
                ) 
                robot_interface.update_robot_model()
                my_world.step(render=True)
                if init_dist < 0.027:
                    grasp_pose = wTgrasp_finder(
                        suc_grasps, robot_interface.robot_model, new_object_pos)
                    pick_position = [grasp_pose.t[0], grasp_pose.t[1], 1.2]
                    current_ori = grasp_pose.R
                    pick_pose = sm.SE3.Rt(current_ori, pick_position, check=False)
                    nav_flag = False
                    pregrasp_flag = True
            if pregrasp_flag:
                current_data = robot_interface.get_camera_data()
                joint_pos = my_robot.get_joint_positions()
                robot_interface.robot_model.fkine(robot_interface.robot_model.q)
                pregrasp_dist, _, ee_twist = move_to_goal(
                    robot_interface, lite_controller, grasp_pose)
                save_rgb_depth(current_data, step_counter, success_counter)
                pose_dict_name = update_pose_dict(
                    pose_dict_name, step_counter, joint_pos, ee_twist)
                robot_interface.update_robot_model()
                my_world.step(render=True)                
                step_counter += 1
                if pregrasp_dist < 0.025:
                    pregrasp_flag = False
                    grasp_flag = True
            if grasp_flag:
                current_data = robot_interface.get_camera_data()
                joint_pos = my_robot.get_joint_positions()
                robot_interface.robot_model.fkine(robot_interface.robot_model.q)
                _, _, ee_twist = move_to_goal(
                    robot_interface, lite_controller, grasp_pose)
                obj_grasped = grasp_obj(robot_interface, lite_controller)
                save_rgb_depth(current_data, step_counter, success_counter)
                pose_dict_name = update_pose_dict(
                    pose_dict_name, step_counter, joint_pos, ee_twist)
                robot_interface.update_robot_model()
                my_world.step(render=True)
                step_counter += 1
                print("step_counter: ", step_counter)
                if obj_grasped:
                    grasp_flag = False
                    pick_flag = True
            if pick_flag:
                current_data = robot_interface.get_camera_data()
                joint_pos = my_robot.get_joint_positions()
                robot_interface.robot_model.fkine(robot_interface.robot_model.q)
                distance_len_f, _, ee_twist = move_to_goal(
                    robot_interface, lite_controller, pick_pose)
                save_rgb_depth(current_data, step_counter, success_counter)
                pose_dict_name = update_pose_dict(
                    pose_dict_name, step_counter, joint_pos, ee_twist)
                robot_interface.close_gripper()
                robot_interface.update_robot_model()
                my_world.step(render=True)
                if my_object.get_world_pose()[0][-1] > 0.98:
                    np.save(
                        "collected_data/traj{}/pose.npy".format(success_counter),
                        pose_dict_name)
                    success_counter += 1
                    pick_flag = False
                    print("successful grasp ", success_counter)
                    break
                if distance_len_f < 0.03:
                    pick_flag = False
                    failure_flag = True
                step_counter += 1
                print("step_counter: ", step_counter)

    simulation_app.close()
    return


if __name__ == "__main__":
    main(nr_traj=100, step_threshold=500)
