import sys
from pathlib import Path
from argparse import ArgumentParser
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R
import spatialmath as sm
from spatialmath.base import trnorm

HOME = str(Path.home())
print("HOME: ", HOME)


def closest_grasp(grasps, ee_pos):
    ee_pos = ee_pos.A
    matrices = grasps - ee_pos
    dists = np.linalg.norm(matrices, axis=(1, 2))
    min_index = np.argmin(dists)
    return grasps[min_index]


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


def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.prims.geometry_prim import GeometryPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
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
    suc_grasps, object_position, object_scale, object_mass = mesh_data(
        config["mesh_dir"]
    )

    print("Number of successful grasps: ", suc_grasps.shape[0])
    print("SCALE: ", object_scale)

    add_reference_to_stage(usd_path=config["cube_dir"], prim_path="/World/cube")

    my_cube = my_world.scene.add(
        GeometryPrim(
            prim_path="/World/cube",
            name="fancy_cube",
            position=suc_grasps[1185][:3, -1],
            scale=[0.0003, 0.0003, 0.002],
            visible=True,
            collision=False,
        )
    )
    my_cube = my_world.scene.add(RigidPrim(prim_path="/World/cube", name="fancy_cube1"))

    add_reference_to_stage(usd_path=config["object_path"], prim_path="/World/bowl")

    my_object = my_world.scene.add(
        GeometryPrim(
            prim_path="/World/bowl",
            name="fancy_bowl",
            position=object_position,
            scale=(object_scale, object_scale, object_scale),
            visible=True,
            collision=True,
        )
    )
    # my_object = my_world.scene.add(
    #     RigidPrim(prim_path="/World/bowl", name="fancy_bowl1", mass=object_mass)
    # )
    # print("CHECK mesh: ", my_world.scene.object_exists(name="fancy_bowl"))

    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    while simulation_app.is_running():
        my_world.step(render=True)


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
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/cup/cup.usd",
        "fps": 50,
        "mesh_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/cup/cup.h5",
        "cube_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/cracker.usd",
    }
    main(config)
