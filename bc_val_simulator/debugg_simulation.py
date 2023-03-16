import sys
import os
from pathlib import Path
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
import spatialmath as sm
from spatialmath.base import trnorm
from debug_helpers import *

HOME = str(Path.home())
print("HOME: ", HOME)


def simulation_main(config, action, ee_info):
    simulation_app = SimulationApp({"headless":False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    import omni.usd
    from isaac_simulator.fmm_isaac import FmmIsaacInterface
    from isaac_simulator.robot_control import ReachLocation

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

    my_world.reset()
    robot_interface = FmmIsaacInterface(robot_sim)
    initial_controller = ReachLocation(robot_interface)

    navigation_flag = False
    pick_and_place_flag = True
    robot = robot_interface.robot_model

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    step_counter = 0
    counter = 0
    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
            observations = get_observations(robot=robot, obj=my_object)

            if (
                pick_and_place_flag
            ):  # One trajectory has been finished (successful or failrue) and a new one should start
                my_world.reset()
                new_object_pos = initial_object_pos_selector()
                gripper_target_pose = gripper_inital_point_selector()
                my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
                navigation_flag = True
                pick_and_place_flag = False
            if (
                navigation_flag
            ):  # Navigate the gripper to the initial pose before starting Pick and Place
                initial_controller.move(gripper_target_pose)
                wTe = robot.fkine(robot.q)
                dist = np.linalg.norm(wTe.t - gripper_target_pose.t)
                if (
                    dist <= 0.025
                ):  # If end effectors are close enough to the target initial pose, find the target pose for Pick and Place
                    wTgrasp = wTgrasp_finder(suc_grasps, robot, new_object_pos)
                    print("Final grasp: ", wTgrasp)
                    navigation_flag = False
            else:  # Pick and Place is performed
                # current_ee_pose = ee_info[step_counter]
                current_ee_pose = robot.fkine(robot.q)
                next_action = action[step_counter]
                target_pose = output_processing(current_ee_pose, next_action)
                initial_controller.move(target_pose)
                if next_action[0,-1] < 0:
                    robot_interface.close_gripper()
                else:
                    robot_interface.open_gripper()
                
                step_counter += 1
                print(step_counter)
                # pick_and_place_flag = True

    simulation_app.close()
    return


if __name__ == "__main__":

    simulation_config = {
        "model_path": HOME + "/Documents/isaac-fmm/models/fmm_full.usd",
        "object_path": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/simple_bowl.usd",
        "fps": 50,
        "mesh_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/bowl.h5",
        "cube_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/cracker.usd",
    }

    traj_dir = "/home/mokhtars/Documents/bc_network/collected_data/traj0"
    ee_info = []
    pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)

    for i in range(len(pose_file.item().keys())):
        ee_info.append(pose_file.item()[i]["ee_pose"])
    image_data, joint_data, action = experience_collector(traj_dir)
    simulation_main(simulation_config, action, ee_info)
    