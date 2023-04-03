import sys
sys.path.append('/home/mokhtars/Documents/bc_network')
import os
from pathlib import Path
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
import spatialmath as sm
from spatialmath.base import trnorm
from utils.helpers import *
from bc_network.bcdataset import ReplayBuffer_abs_rot as buffer
import time
import cv2
import yaml
from yaml.loader import SafeLoader

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
    from fmm_old_control.fmm_old_isaac import FmmIsaacInterface
    from fmm_old_control.robot_control import ReachLocation

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit()

    # Initialize World
    world_settings = {
        "stage_units_in_meters": 1.0
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
    target_ee_pose = {}
    actual_ee_pose = {}
    get_current_ee = {}
    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    step_counter = 0
    counter = 0
    my_world.set_simulation_dt(physics_dt=1.0 / 40.0, rendering_dt=1.0 / 40.0)
    while simulation_app.is_running():
        start_time = time.time()
        my_world.step(render=True)
        print("Time step index", my_world.current_time_step_index)
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
            if step_counter == 205:
                break
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
                current_ee_pose = ee_info[step_counter]

                # current_data = robot_interface.get_camera_data()
                # rgb_image = cv2.cvtColor(current_data["rgb"][:, :, 0:3], cv2.COLOR_BGR2RGB)
                # cv2.imwrite("collected_data/traj{}/rgb/{}.png".format(10,
                #         step_counter), rgb_image)
                # cv2.imwrite("collected_data/traj{}/depth/{}.png".format(10,
                #         step_counter), current_data["depth"])

                # current_ee_pose = robot.fkine(robot.q)
                next_action = action[step_counter]
                # next_action[0][0:3] = torch.tensor(ee_info[step_counter+1].t - ee_info[step_counter].t)
                target_pose = output_processing(current_ee_pose, next_action)
                # target_ee_pose[step_counter] = target_pose
                initial_controller.move(current_ee_pose)
                get_current_ee[step_counter] = robot.fkine(robot.q)
                if next_action[0,-1] < 0:
                    robot_interface.close_gripper()
                else:
                    robot_interface.open_gripper()
                # actual_ee_pose[step_counter] = robot.fkine(robot.q)
                step_counter += 1
                print(step_counter)
                # pick_and_place_flag = True

        end_time = time.time()
        # Calculate actual FPS
        fps = 1 / (end_time - start_time)
        print("actual fps: ", fps)
    # np.save("/home/mokhtars/Documents/bc_network/target_ee_pose.npy", target_ee_pose)
    # np.save("/home/mokhtars/Documents/bc_network/actual_ee_pose.npy", actual_ee_pose)
    np.save("/home/mokhtars/Documents/bc_network/corresponding_ee_pose.npy", get_current_ee)
    simulation_app.close()
    return


if __name__ == "__main__":

    with open('simulation_config.yaml') as f:
        simulation_config = yaml.load(f, Loader=SafeLoader)
        print("simulation config: ", simulation_config)

    with open('config.yaml') as f:
        network_config = yaml.load(f, Loader=SafeLoader)
        print("network config: ", network_config)
        
    data_dir = "/home/mokhtars/Documents/bc_network/collected_data"
    traj_dir = "/home/mokhtars/Documents/bc_network/collected_data/traj0_old"
    ee_info = []
    pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)

    for i in range(len(pose_file.item().keys())):
        ee_info.append(pose_file.item()[i]["ee_pose"])
    replay_memory = buffer(network_config["buffer_capacity"], data_dir, network_config["sequence_len"])
    image_data, joint_data, action = replay_memory.experience_collector(traj_dir)
    simulation_main(simulation_config, action, ee_info)
    