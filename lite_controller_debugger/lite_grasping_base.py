import sys
sys.path.append('/home/mokhtars/Documents/bc_network')
from pathlib import Path
import pathlib
import time
import yaml
from yaml.loader import SafeLoader
import numpy as np
import spatialmath as sm
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
    from omni.isaac.core.prims.geometry_prim import GeometryPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils  # type: ignore
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
    # shape_approximation2 = "convexHull"
    # shape_approximation3 = "none"
    # shape_approximation4 = "meshSimplification"
    # shape_approximation5 = "convexMeshSimplification"
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

    obj_grasped = False
    my_world.step(render=True)
    while simulation_app.is_running():
        # Move to pose 1
        # pose1 = EE_POSE
        new_object_pos = initial_object_pos_selector()
        gripper_target_pose = gripper_inital_point_selector()
        my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
        pose1 = gripper_target_pose
        init_dist = 1
        while init_dist > 0.01:
            print("my_world.current_time_step_index", my_world.current_time_step_index)
            init_dist, goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, pose1
            )
            robot_interface.update_robot_model()
            my_world.step(render=True)
            # time.sleep(1.0)
        print("Reached goal")

        # Move to pose 2
        # pose2 = sm.SE3.Ty(0.5) * pose1
        pose2 = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)
        print("Final grasp: ", pose2)
        sec_dist = 1
        while sec_dist > 0.02:
            print("my_world.current_time_step_index", my_world.current_time_step_index)
            grTpregr = sm.SE3.Trans(0.0, 0.0, -0.05)
            wTgr = pose2
            wTpregr = sm.SE3(wTgr.A @ grTpregr.A, check=False)
            sec_dist, goal_reached, ee_twist = move_to_goal(robot_interface, lite_controller, wTpregr)
            robot_interface.update_robot_model()
            my_world.step(render=True)
            print("distance_len: ", sec_dist)
        print("Moved to pregrasp")
        # gt = robot_interface.get_camera_data()
        while not obj_grasped:
            gt = robot_interface.get_camera_data()
            print("my_world.current_time_step_index", my_world.current_time_step_index)
            distance_len, goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, pose2
            )
            if distance_len < 0.025:
                obj_grasped = grasp_obj(robot_interface, lite_controller)
            robot_interface.update_robot_model()
            my_world.step(render=True)
            # print("distance_len: ", distance_len)
     
        desired_pos = [pose2.t[0], pose2.t[1], 0.9]
        # vertical_grasp = sm.SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        current_ori = pose2.R
        pose3 = sm.SE3.Rt(current_ori, desired_pos, check=False)
        while my_object.get_world_pose()[0][-1] < 0.8:
            print("my_world.current_time_step_index", my_world.current_time_step_index)
            distance_len_f, goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, pose3
            )
            robot_interface.close_gripper()
            robot_interface.update_robot_model()
            # print("distance_len_f: ", distance_len_f)
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
    
    return distance_lin, goal_reached, ee_twist

def grasp_obj(robot_interface, controller):
    robot_interface.close_gripper()
    # robot_interface.update_robot_model()
    controller.grasp_counter += 1
    obj_grasped = True if controller.grasp_counter > 70 else False
    return obj_grasped


if __name__ == "__main__":

    with open('simulation_config.yaml') as f:
        simulation_config = yaml.load(f, Loader=SafeLoader)
        print("simulation config: ", simulation_config)
    main(simulation_config)
