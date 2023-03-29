import sys
sys.path.append('/home/mokhtars/Documents/bc_network/fmm-control-lite/fmm_control_lite')
import os
from pathlib import Path
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
import time

HOME = str(Path.home())
print("HOME: ", HOME)

class SteadyRate:
    """ Maintains the steady cycle rate provided on initialization by adaptively sleeping an amount
    of time to make up the remaining cycle time after work is done.

    Usage:

    rate = SteadyRate(rate_hz=60.)
    while True:
      app.update() # render/app update call here
      rate.sleep()  # Sleep for the remaining cycle time.

    """
    def __init__(self, rate_hz):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.last_sleep_end = time.time()

    def sleep(self):
        work_elapse = time.time() - self.last_sleep_end
        sleep_time = self.dt - work_elapse
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        self.last_sleep_end = time.time()


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
    area == 2  #comment this line for random initial pose of gripper
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
    elif area == 2:
        lower = np.array([1.0, 0.20, 1.18])
        upper = np.array([1.0, 0.20, 1.18])
        theta = -50
        r = R.from_euler("xyz", [180, theta, 0], degrees=True)
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
    # if area == 0 then the initial pose of object is random, 
    # if area ==2 then the initial pose of object is fixed
    area = random.randint(2, 2)    

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
    elif area == 2:
        x = np.random.uniform(low=1.7, high=1.7, size=1)
        y = np.random.uniform(low=0.2, high=0.2, size=1)
        z = np.random.uniform(low=0.6, high=0.6, size=1)
        object_pose = [x[0], y[0], z[0]]
    else:
        x = np.random.uniform(low=-1.4, high=-1.05, size=1)
        y = np.random.uniform(low=-0.05, high=0.45, size=1)
        z = np.random.uniform(low=0, high=0, size=1)
        object_pose = [x[0], y[0], z[0]]

    return area, object_pose


def grasp_obj(robot_interface, grasp_counter):
    robot_interface.close_gripper()
    grasp_counter += 1
    obj_grasped = True if grasp_counter > 20 else False
    return obj_grasped

def pick_pose(robot):
    last_ee_pose = robot.fkine(robot.q)
    desired_pos = last_ee_pose.t + [0, 0, 0.3]
    vertical_grasp = sm.SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    postgrasp_pose = sm.SE3.Rt(vertical_grasp, desired_pos)
    return postgrasp_pose

def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    import omni.usd
    from fmm_isaac import FmmIsaacInterface_lite
    from fmm_control import FmmQPControl


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
    robot_interface = FmmIsaacInterface_lite(robot_sim)
    robot = robot_interface.robot_model
    lite_controller = FmmQPControl(dt=(1.0 / config["fps"]), fmm_mode="no-tower", robot=robot, robot_interface=robot_interface)

    navigation_flag = False
    pick_and_place_flag = True
    grasp_flag = True

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    nr_traj = 1

    success_counter = 0
    failure_counter = 0
    step_counter = 0

    #-----sleep-----
    # rate = SteadyRate(config["fps"])
    #---------------
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
                area, new_object_pos = initial_object_pos_selector()
                gripper_target_pose = gripper_inital_point_selector(area)
                my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
                navigation_flag = True
                pick_and_place_flag = False
                step_counter = 0
                grasp_counter = 0
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
                
                initial_qd, initial_distance = lite_controller.wTeegoal_2_qd(gripper_target_pose)
                print("initial_distance: ", initial_distance)

                if (
                    initial_distance <= 0.02
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
                    graspTgoal[2, 3] = 0.1234

                    graspTgoal = sm.SE3(graspTgoal)
                    wTgrasp = wTobj * objTgrasp * graspTgoal * fT
                    print("Final grasp: ", wTgrasp)
                    navigation_flag = False

            else:  # Pick and Place is performed
                current_data = robot_interface.get_camera_data()
                rgb_im = Image.fromarray(current_data["rgb"][:, :, 0:3])
                rgb_im.save(
                    "collected_data/traj{}/rgb/{}.jpeg".format(
                        success_counter, step_counter
                    )
                )
                depth_im = current_data["depth"]
                depth_im = (depth_im * 100.0).astype(np.uint16)
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

                if grasp_flag:
                    qd, distance = lite_controller.wTeegoal_2_qd(wTgrasp)
                    # print("gripper_position: ", robot_sim.get_joint_positions()[-2:])
                    print("Grasp distance: ", distance)
                    if distance <= 0.015:
                        obj_grasped = grasp_obj(robot_interface, grasp_counter)
                        if obj_grasped:
                            grasp_flag = False
                else:
                    postgrasp_pose = pick_pose(robot)
                    post_qd, final_distance = lite_controller.wTeegoal_2_qd(postgrasp_pose)
                    print("Final distance: ", final_distance)
                    if final_distance <= 0.012:
                        object_height = my_object.get_world_pose()[0][-1]
                        print("Object_height: ", object_height)
                        if object_height > 0.65:  # success
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
                      
                step_counter += 1
                print("step: ", step_counter)
                if step_counter == 300:  # failure
                    print("FAILURE")
                    shutil.rmtree(traj_path)
                    failure_counter += 1
                    pick_and_place_flag = True

        end_time = time.time()
        # rate.sleep()
        # Calculate actual FPS
        fps = 1 / (end_time - start_time)
        print("actual fps: ", fps)

    simulation_app.close()
    return


if __name__ == "__main__":
    config = {
        "model_path": HOME + "/Documents/isaac-fmm/models/fmm_full.usd",
        "object_path": HOME
        + "/Documents/bc_network/data_collector/imitation_learning/bowl/simple_bowl.usd",
        "fps": 15,
        "mesh_dir": HOME
        + "/Documents/bc_network/data_collector/imitation_learning/bowl/bowl.h5",
        "cube_dir": HOME
        + "/Documents/bc_network/data_collector/imitation_learning/cracker.usd",
    }
    main(config)
