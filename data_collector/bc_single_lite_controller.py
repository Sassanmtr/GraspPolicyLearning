import sys
sys.path.append('/home/mokhtars/Documents/bc_network')
from pathlib import Path
import pathlib
import time
import yaml
from yaml.loader import SafeLoader
import numpy as np
import spatialmath as sm
import shutil
from omni.isaac.kit import SimulationApp
from utils.helpers import *

HOME = str(Path.home())
print("HOME: ", HOME)

def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.physx.scripts import utils 
    import omni.usd
    from src.fmm_isaac import FmmIsaacInterface
    from fmm_control_lite.fmm_control import FmmQPControl

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

    # Initialize Env and Object
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

    print("Object mass: ", object_mass)
    print("Adding PHYSICS to ShapeNet model")
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim("/World/Hospital/object", "Xform")
    shape_approximation = "convexDecomposition"
    utils.setRigidBody(prim, shape_approximation, False)
    # utils.setCollider(prim, approximationShape=shape_approximation)
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

    nr_traj = 100
    step_threshold = 500
    success_counter = 0
    failure_counter = 0
    failure_flag = False
    obj_grasped = False
    while simulation_app.is_running():
        my_world.reset()
        my_world.step(render=True)
        new_object_pos = initial_object_pos_selector()
        ee_initial_target_pose = gripper_inital_point_selector()
        my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
        grasp_pose = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)
        
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
            # print("step counter: ", step_counter)
            if failure_flag or step_counter == step_threshold:
                shutil.rmtree(traj_path)
                failure_counter += 1
                print("Failure ", failure_counter)
                failure_flag = False
                break
            if nav_flag:
                init_dist, goal_reached, ee_twist = move_to_goal(
                    robot_interface, lite_controller, ee_initial_target_pose
                ) 
                robot_interface.update_robot_model()
                robot_interface.get_camera_data()
                my_world.step(render=True)
                if init_dist < 0.027:
                    grasp_pose = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)
                    pick_position = [grasp_pose.t[0], grasp_pose.t[1], 1.2]
                    current_ori = grasp_pose.R
                    pick_pose = sm.SE3.Rt(current_ori, pick_position, check=False)
                    nav_flag = False
                    pregrasp_flag = True
            if pregrasp_flag:
                # print("pregrasp...")
                pregrasp_dist, goal_reached, ee_twist = move_to_goal(robot_interface, lite_controller, grasp_pose)
                robot_interface.update_robot_model()
                my_world.step(render=True)
                # Save RGB and Depth images
                print("my_world.current_time_step_index", my_world.current_time_step_index)
                # print("pregrasp dist: ", pregrasp_dist)
                current_data = robot_interface.get_camera_data()
                save_rgb_depth(current_data, step_counter, success_counter)
                pose_dict_name = update_pose_dict(pose_dict_name, step_counter, robot_sim, ee_twist)
                
                step_counter += 1
                
                if pregrasp_dist < 0.027:
                    pregrasp_flag = False
                    grasp_flag = True
            if grasp_flag:
                # print("grasp...")
                distance_len, goal_reached, ee_twist = move_to_goal(
                    robot_interface, lite_controller, grasp_pose)
                obj_grasped = grasp_obj(robot_interface, lite_controller)
                robot_interface.update_robot_model()
                my_world.step(render=True)
                # Save RGB and Depth images and update pose dictionary
                current_data = robot_interface.get_camera_data()
                save_rgb_depth(current_data, step_counter, success_counter)
                pose_dict_name = update_pose_dict(pose_dict_name, step_counter, robot_sim, ee_twist)
                step_counter += 1
                print("step_counter: ", step_counter)
                if obj_grasped:
                    grasp_flag = False
                    pick_flag = True

            if pick_flag:
                # print("pick...")
                distance_len_f, goal_reached, ee_twist = move_to_goal(
                    robot_interface, lite_controller, pick_pose)
                robot_interface.close_gripper()
                robot_interface.update_robot_model()
                my_world.step(render=True)
                # print("object distance: ", distance_len_f)
                # print("my_world.current_time_step_index", my_world.current_time_step_index)
                # Save RGB and Depth images and update pose dictionary
                current_data = robot_interface.get_camera_data()
                save_rgb_depth(current_data, step_counter, success_counter)
                pose_dict_name = update_pose_dict(pose_dict_name, step_counter, robot_sim, ee_twist)
        
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


if __name__ == "__main__":

    with open('simulation_config.yaml') as f:
        simulation_config = yaml.load(f, Loader=SafeLoader)
        print("simulation config: ", simulation_config)
    main(simulation_config)
