import sys
sys.path.append('/home/mokhtars/Documents/bc_network')
import pathlib
import time
import numpy as np
import yaml
from yaml.loader import SafeLoader
import spatialmath as sm
from utils.helpers import *
from omni.isaac.kit import SimulationApp


def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.xform_prim_view import XFormPrimView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils  # type: ignore
    import omni.usd
    from omni.isaac.synthetic_utils import SyntheticDataHelper
    from omni.kit.viewport_legacy import get_viewport_interface
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
    # Initialize Env
    add_reference_to_stage(
        usd_path="home/mokhtars/Documents/isaac-fmm/models/hospital.usd",
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

    EE_POSE = sm.SE3(
        np.array(
            [
                [0.84132622, -0.00711519, 0.54048087, 1.46307772],
                [0.00374581, -0.9998126, -0.01899292, -0.13671148],
                [0.54051472, 0.01800378, -0.8411419, 0.979675],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        check=False,
    )
    my_world.step(render=True)
    # Detecting the camera
    my_camera = my_world.scene.add(
        XFormPrimView(
            prim_paths_expr="/World/FMM/wrist_camera/Camera",
            name="fancy_camera",
        )
    )
    print("CHECK camera: ", my_world.scene.object_exists(name="fancy_camera"))
    print(my_camera.get_world_poses()[0])
    viewport_interface = get_viewport_interface()
    viewport_handle = viewport_interface.create_instance()
    viewport_window = viewport_interface.get_viewport_window(viewport_handle)
    viewport_window.set_active_camera("/World/FMM/wrist_camera/Camera")
    viewport_window.set_texture_resolution(500, 300)
    viewport_window.set_window_pos(500, 500)
    viewport_window.set_window_size(500, 300)
    data_helper = SyntheticDataHelper()
    #  Camera matrix parameters
    width = 512
    height = 300
    focal_length = 1.93
    horiz_aperture = 2.682
    vert_aperture = 1.509
    fx = height * focal_length / vert_aperture
    fy = width * focal_length / horiz_aperture
    x0 = height * 0.5
    y0 = width * 0.5

    while simulation_app.is_running():
        # Move to pose 1
        new_object_pos = initial_object_pos_selector()
        gripper_target_pose = gripper_inital_point_selector()
        my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
        pose1 = EE_POSE
        goal_reached = False
        while not goal_reached:
            print("my_world.current_time_step_index", my_world.current_time_step_index)
            gt = data_helper.get_groundtruth(
                sensor_names=["rgb"],
                viewport=viewport_window,
                wait_for_sensor_data=0.1
            )
            # gt = data_helper.get_pose()
            gt = data_helper.get_semantic_ids()
            goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, pose1
            )
            robot_interface.update_robot_model()
            my_world.step(render=True)
            # time.sleep(1.0)
        print("Reached goal")

        # Move to pose 2
        pose2 = sm.SE3.Ty(0.5) * pose1
        goal_reached = False
        while not goal_reached:
            goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, pose2
            )
            robot_interface.update_robot_model()
            my_world.step(render=True)
            # time.sleep(1.0)
        print("Reached goal")

        # Exit simulation
        break

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
    return goal_reached, ee_twist


if __name__ == "__main__":
    with open('simulation_config.yaml') as f:
        simulation_config = yaml.load(f, Loader=SafeLoader)
        print("simulation config: ", simulation_config)
    main(simulation_config)
