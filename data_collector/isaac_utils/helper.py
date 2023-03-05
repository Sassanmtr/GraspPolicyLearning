import sys
import os
import numpy as np
import spatialmath as sm
from spatialmath.base import trnorm
from pathlib import Path
from omni.isaac.kit import SimulationApp
import carb
from pxr import Gf
from scipy.spatial.transform import Rotation as R

HOME = str(Path.home())


def scene_contacts_reader(idx=0):
    print(os.getcwd())
    scene_contacts_dir = os.getcwd() + "/imitation_learning/Scene_contacts"
    scene_contacts_files = os.listdir(scene_contacts_dir)
    scene_contacts_files.sort()
    print(scene_contacts_files)
    selected_scene = scene_contacts_files[idx]
    path = os.path.join(scene_contacts_dir, selected_scene)
    print("Reading: ", path)
    file = np.load(path, allow_pickle=True)
    obj_paths = file["obj_paths"]
    obj_transforms = file["obj_transforms"]
    obj_scale = file["obj_scales"]
    grasp_transforms = file["grasp_transforms"]
    scene_contact_points = file["scene_contact_points"]
    obj_grasp_idcs = file["obj_grasp_idcs"]
    return obj_paths, obj_transforms, obj_scale, grasp_transforms, obj_grasp_idcs


def closest_grasp(grasps, robot):
    ee_pos = robot.fkine(robot.q)
    ee_pos = ee_pos.A
    matrices = grasps - ee_pos
    dists = np.linalg.norm(matrices, axis=(1, 2))
    min_index = np.argmin(dists)
    return grasps[min_index]


def closest_object(obj_poses, robot):
    ee_pose = robot.fkine(robot.q)
    ee_pos = ee_pose.A
    distances = []
    for i in range(len(obj_poses)):
        distances.append(np.linalg.norm(ee_pos[:3, -1] - obj_poses[i][0]))
    min_index = np.argmin(np.array(distances))
    return min_index


def gripper_inital_point_selector(area):
    if area == 0:
        lower = np.array([1.05, 0.21, 1.19])
        upper = np.array([0.95, 0.18, 1.17])
        theta = -50
        r = R.from_euler("xyz", [180, theta, 0], degrees=True)
    x = np.random.uniform(low=lower[0], high=upper[0], size=1)
    y = np.random.uniform(low=lower[1], high=upper[1], size=1)
    z = np.random.uniform(low=lower[2], high=upper[2], size=1)
    target_ori = sm.SO3(np.array(r.as_matrix()))
    target_pos = [x[0], y[0], z[0]]
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose


def quater_rotation(quatern):
    quaternion = quatern.reshape(
        4,
    )
    gf_quaternion = Gf.Quatf(
        float(quaternion[0]),
        Gf.Vec3f(float(quaternion[1]), float(quaternion[2]), float(quaternion[3])),
    )
    mat = Gf.Matrix3d(gf_quaternion).GetTranspose()
    return mat


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
        usd_path=HOME + "/Documents/Isaac-environments/Collected_hospital/hospital.usd",
        prim_path="/World/Hospital",
    )
    my_world.scene.add(
        XFormPrim(prim_path="/World/Hospital", name="hospital", position=[0, 0, 0])
    )
    initial_object_poses = []
    (
        obj_paths,
        obj_transforms,
        obj_scale,
        grasp_transforms,
        obj_grasp_idcs,
    ) = scene_contacts_reader(0)

    objects = []
    paths = []
    for i, obj_path in enumerate(obj_paths):
        object_path = os.path.join(config["meshes_dir"], obj_path)
        object_path = object_path[:-3] + "usd"
        paths.append(object_path)
        add_reference_to_stage(
            usd_path=object_path, prim_path="/World/Hospital/object{}".format(i)
        )
        object_name = "my_object{}".format(i)
        object_name = my_world.scene.add(
            XFormPrim(
                prim_path="/World/Hospital/object{}".format(i),
                name="fancy_object{}".format(i),
                position=obj_transforms[i][:3, -1] + np.array([1, 0, 0]),
                scale=(obj_scale[i], obj_scale[i], obj_scale[i]),
                visible=True,
            )
        )
        print("Adding PHYSICS to ShapeNet model")
        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim("/World/Hospital/object{}".format(i), "Xform")
        shape_approximation = "convexDecomposition"
        utils.setRigidBody(prim, shape_approximation, False)
        print(
            "CHECK mesh{}: ".format(i),
            my_world.scene.object_exists(name="fancy_object{}".format(i)),
        )
        objects.append(object_name)
        initial_object_poses.append(object_name.get_world_pose())

    my_world.reset()
    robot_interface = FmmIsaacInterface(robot_sim)
    initial_controller = ReachLocation(robot_interface)
    final_controller = PickAndPlace(robot_interface)

    navigation_flag = True
    pick_and_place_flag = False
    robot = robot_interface.robot_model

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()

    nr_traj = 10
    success_counter = 0
    failure_counter = 0
    step_counter = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
            if failure_counter >= nr_traj * 3:
                print("FAILURE, not enough successful grasps")
                break
            if (
                success_counter == nr_traj
            ):  # Enough successful trajectories have been recorded
                break
            if navigation_flag:
                gripper_target_pose = gripper_inital_point_selector(area=0)
                initial_controller.move(gripper_target_pose)
                wTe = robot.fkine(robot.q)
                dist = np.linalg.norm(wTe.t - gripper_target_pose.t)
                if (
                    dist <= 0.012
                ):  # If end effectors are close enough to the target initial pose
                    current_objects_pose = []
                    for i in range(len(obj_paths)):
                        current_objects_pose.append(objects[i].get_world_pose())
                    object_index = closest_object(current_objects_pose, robot)
                    print(object_index)
                    print(paths[object_index])
                    if object_index == 0:
                        possible_grasps = grasp_transforms[
                            0 : obj_grasp_idcs[object_index]
                        ]
                    else:
                        possible_grasps = grasp_transforms[
                            obj_grasp_idcs[object_index - 1]
                            + 1 : obj_grasp_idcs[object_index]
                        ]
                    selected_grasp = closest_grasp(grasps=possible_grasps, robot=robot)
                    print("Selected grasp: ", selected_grasp)
                    wTobj = sm.SE3.Rt(
                        sm.SO3(
                            trnorm(
                                np.array(
                                    quater_rotation(
                                        objects[object_index].get_world_pose()[1]
                                    )
                                )
                            )
                        ),
                        objects[object_index].get_world_pose()[0],
                    )
                    objTgrasp = sm.SE3(trnorm(selected_grasp))
                    # objTobj = np.eye(4)
                    # objTobj[:3, -1] = (
                    #     -objects[object_index].get_world_pose()[0]
                    #     + initial_object_poses[object_index][0]
                    # )
                    # objTobj = sm.SE3(objTobj)
                    graspTgoal = np.eye(4)
                    fT = np.array(
                        [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                    )
                    fT = sm.SE3(fT)
                    graspTgoal[2, 3] = 0.1034

                    graspTgoal = sm.SE3(graspTgoal)
                    wTgrasp = wTobj * objTgrasp * graspTgoal * fT
                    print("Final grasp: ", wTgrasp)
                    navigation_flag = False
                    pick_and_place_flag = True
            if pick_and_place_flag:
                final_controller.move(
                    wTgrasp,
                    sm.SE3.Rt(
                        sm.SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])),
                        [0.6, -1.5, 0.7],
                    ),
                )


# ----------------------------------------------------------------

if __name__ == "__main__":

    config = {
        "model_path": HOME + "/Documents/isaac-fmm/models/fmm_full.usd",
        "meshes_dir": os.getcwd() + "/imitation_learning",
        "object_path": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/simple_bowl.usd",
        "fps": 50,
        "mesh_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/bowl.h5",
        "cube_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/cracker.usd",
    }
    main(config)
