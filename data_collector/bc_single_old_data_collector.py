import sys
sys.path.append('/home/mokhtars/Documents/bc_network')
import os
from pathlib import Path
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
import cv2
import shutil
import time
import yaml
from yaml.loader import SafeLoader
from utils.helpers import *

HOME = str(Path.home())
print("HOME: ", HOME)

def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    import omni.usd
    from fmm_old_control.fmm_old_isaac import FmmIsaacInterface
    from fmm_old_control.robot_control import PickAndPlace, ReachLocation

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit()

    # Initialize World
    world_settings = {
        "stage_units_in_meters": 1.0,
    }
    my_world = World(**world_settings)
    # my_world._time_steps_per_second = config["fps"]
    # my_world._fsm_update_rate = config["fps"]
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
    print("Number of successful grasps: ", suc_grasps.shape[0])
    print("SCALE: ", object_scale)

    add_reference_to_stage(
        usd_path=config["object_path"], prim_path="/World/Hospital/object"
    )

    my_object = my_world.scene.add(
        XFormPrim(
            prim_path="/World/Hospital/object",
            name="fancy_object",
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
    print("CHECK mesh: ", my_world.scene.object_exists(name="fancy_object"))

    my_world.reset()
    robot_interface = FmmIsaacInterface(robot_sim)
    initial_controller = ReachLocation(robot_interface)
    final_controller = PickAndPlace(robot_interface)

    navigation_flag = False
    pick_and_place_flag = True
    robot = robot_interface.robot_model

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()

    nr_traj = 1

    success_counter = 0
    failure_counter = 0
    step_counter = 0
    my_world.set_simulation_dt(physics_dt=1.0 / 40.0, rendering_dt=1.0 / 10000.0)
    while simulation_app.is_running():
        start_time = time.time()
        my_world.step(render=False)
        print("Time step index", my_world.current_time_step_index)
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
            observations = get_observations(robot=robot, obj=my_object)
            if failure_counter >= nr_traj * 3:
                print("FAILURE, not enough successful grasps")
                break
            
            if (
                success_counter == nr_traj
            ):  # Enough successful trajectories have been recorded
                break
            
            if (
                pick_and_place_flag
            ):  # One trajectory has been finished (successful or failrue) and a new one should be started
                my_world.reset()
                new_object_pos = initial_object_pos_selector()
                gripper_target_pose = gripper_inital_point_selector()
                my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
                final_controller.reset()
                navigation_flag = True
                pick_and_place_flag = False
                step_counter = 0
                traj_path = os.path.join(
                    os.getcwd() + "/collected_data",
                    "traj{}".format(success_counter),
                )
                os.mkdir(traj_path)
                rgb_path = os.path.join(
                    os.getcwd() + "/collected_data",
                    "traj{}/rgb".format(success_counter),
                )
                os.mkdir(rgb_path)
                depth_path = os.path.join(
                    os.getcwd() + "/collected_data",
                    "traj{}/depth".format(success_counter),
                )
                os.mkdir(depth_path)

                pose_dict_name = "pose{}".format(success_counter)
                pose_dict_name = {}
            if (
                navigation_flag
            ):  # Navigate the gripper to the initial pose before starting Pick and Place
                initial_controller.move(gripper_target_pose)
                wTe = robot.fkine(robot.q)
                dist = np.linalg.norm(wTe.t - gripper_target_pose.t)
                if (
                    dist <= 0.012
                ):  # If end effectors are close enough to the target initial pose
                    print("Gripper is in the initial grasping pose!")
                    wTgrasp = wTgrasp_finder(suc_grasps, robot, new_object_pos)
                    navigation_flag = False
            else:  # Pick and Place is performed
                if final_controller.save_mode:
                    current_data = robot_interface.get_camera_data()
                    rgb_image = cv2.cvtColor(current_data["rgb"][:, :, 0:3], cv2.COLOR_BGR2RGB)
                    cv2.imwrite("collected_data/traj{}/rgb/{}.png".format(success_counter,
                         step_counter), rgb_image)
                    cv2.imwrite("collected_data/traj{}/depth/{}.png".format(success_counter,
                         step_counter), current_data["depth"])

                    pose_dict_name[step_counter] = {}
                    pose_dict_name[step_counter]["ee_pose"] = robot.fkine(robot.q)
                    pose_dict_name[step_counter][
                        "joint_pos"
                    ] = robot_sim.get_joint_positions()

                final_controller.move(
                    wTgrasp, observations[my_object.name]["target_position"]
                )
                # print("gripper_position: ", robot_sim.get_joint_positions()[-2:])
                object_height = my_object.get_world_pose()[0][-1]
                print("Object_height: ", object_height)
                if final_controller.done or object_height > 0.70:  # success
                    if object_height > 0.70:
                        np.save(
                            "collected_data/traj{}/pose.npy".format(success_counter),
                            pose_dict_name,
                        )
                        success_counter += 1
                        print("success!")
                    else:
                        failure_counter += 1
                        shutil.rmtree(traj_path)
                        print("Nope!")
                    pick_and_place_flag = True

                    print("success_counter: ", success_counter)
                step_counter += 1
                print("step: ", step_counter)
                if step_counter == 250:  # failure
                    print("FAILURE")
                    shutil.rmtree(traj_path)
                    failure_counter += 1
                    pick_and_place_flag = True
                                
        end_time = time.time()
        # Calculate actual FPS
        fps = 1 / (end_time - start_time)
        print("actual fps: ", fps)
    simulation_app.close()
    return


if __name__ == "__main__":

    with open('simulation_config.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
        print("config: ", config)
    main(config)
