import sys
import os
from pathlib import Path
from argparse import ArgumentParser
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
import random
import h5py
from scipy.spatial.transform import Rotation as R
import spatialmath as sm
from spatialmath.base import trnorm
from PIL import Image
import cv2
import shutil

HOME = str(Path.home())
print("HOME: ", HOME)


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


def gripper_inital_point_selector(area):
    if area == 0:
        lower = np.array([1.05, 0.21, 1.19])
        upper = np.array([0.95, 0.18, 1.17])
        # y_rot = np.array([-33, -27])
        # theta = np.random.randint(low=y_rot[0], high=y_rot[1], size=1)
        # r = R.from_euler("xyz", [180, theta[0], 0], degrees=True)
        theta = -50
        r = R.from_euler("xyz", [180, theta, 0], degrees=True)
    elif area == 1:
        lower = np.array([0.2, 0.7, 0.9])
        upper = np.array([0.2, 0.7, 0.9])
        theta = 40
        r = R.from_euler("xyz", [180, theta, 90], degrees=True)
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


def closest_grasp(grasps, ee_pos):
    ee_pos = ee_pos.A
    matrices = grasps - ee_pos
    dists = np.linalg.norm(matrices, axis=(1, 2))
    min_index = np.argmin(dists)
    return grasps[min_index]


def initial_object_pos_selector():
    area = random.randint(0, 0)
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
    else:
        x = np.random.uniform(low=-1.4, high=-1.05, size=1)
        y = np.random.uniform(low=-0.05, high=0.45, size=1)
        z = np.random.uniform(low=0, high=0, size=1)
        object_pose = [x[0], y[0], z[0]]

    return area, object_pose


def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    import omni.usd
    from fmm_isaac import FmmIsaacInterface
    from robot_control import PickAndPlace, ReachLocation

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
    print("Number of successful grasps: ", suc_grasps.shape[0])
    print("SCALE: ", object_scale)

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
    # target_pose = gripper_inital_point_selector()
    final_controller = PickAndPlace(robot_interface)

    navigation_flag = False
    pick_and_place_flag = True
    robot = robot_interface.robot_model

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    nr_traj = 3

    success_counter = 0
    failure_counter = 0
    step_counter = 0

    while simulation_app.is_running():
        my_world.step(render=True)
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
                area, new_object_pos = initial_object_pos_selector()
                gripper_target_pose = gripper_inital_point_selector(area)
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
                    selected_grasp = closest_grasp(
                        grasps=suc_grasps,
                        ee_pos=observations[robot.name]["end_effector_position"],
                    )
                    print("Selected grasp: ", selected_grasp)
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
                    graspTgoal[2, 3] = 0.1434

                    graspTgoal = sm.SE3(graspTgoal)
                    wTgrasp = wTobj * objTgrasp * graspTgoal * fT
                    print("Final grasp: ", wTgrasp)
                    navigation_flag = False
            else:  # Pick and Place is performed
                if final_controller.save_mode:
                    current_data = robot_interface.get_camera_data()
                    rgb_im = Image.fromarray(current_data["rgb"][:, :, 0:3])
                    rgb_im.save(
                        "collected_data/traj{}/rgb/{}.jpeg".format(
                            success_counter, step_counter
                        )
                    )
                    depth_im = current_data["depth"]
                    depth_im = (depth_im * 256.0).astype(np.uint16)
                    cv2.imwrite(
                        "collected_data/traj{}/depth/{}.jpeg".format(
                            success_counter, step_counter
                        ),
                        depth_im,
                    )
                    pose_dict_name[step_counter] = {}
                    pose_dict_name[step_counter]["ee_pose"] = robot.fkine(robot.q)
                    pose_dict_name[step_counter][
                        "joint_pos"
                    ] = robot_sim.get_joint_positions()

                final_controller.move(
                    wTgrasp, observations[my_object.name]["target_position"]
                )

                if final_controller.done == True:  # success
                    target_dist = np.linalg.norm(
                        np.array(my_object.get_world_pose()[0])
                        - observations[my_object.name]["target_position"].t
                    )
                    print("Target_dist: ", target_dist)
                    if target_dist <= 0.25:
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
                if step_counter == 1000:  # failure
                    print("FAILURE")
                    shutil.rmtree(traj_path)
                    failure_counter += 1
                    pick_and_place_flag = True

    simulation_app.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-nl",
        "--no-livestream",
        dest="no_livestream",
        action="store_true",
        help="Set this flag if you don't want to livestream from server",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="DemoControl",
        help="options: DemoControl, MoveContainer",
    )
    args = parser.parse_args()
    # if not args.no_livestream:
    #     sys.argv.append("--enable")
    #     sys.argv.append("omni.kit.livestream.native")
    #     sys.argv.append("--no-window")
    #     sys.argv.append("--/app/livestream/allowResize=true")
    config = {
        "task": args.task,
        "model_path": HOME + "/Documents/isaac-fmm/models/fmm_full.usd",
        "object_path": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/simple_bowl.usd",
        "fps": 50,
        "mesh_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/bowl.h5",
        "cube_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/cracker.usd",
    }
    main(config)
