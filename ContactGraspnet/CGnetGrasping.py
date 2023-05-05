import sys
sys.path.append('/home/mokhtars/Documents/bc_network')
from omni.isaac.kit import SimulationApp
from graspnet.contact_graspnet.inference import inference
from pathlib import Path
import numpy as np
from utils.helpers import *
from pxr import Gf
# import argparse
# from graspnet.contact_graspnet import config_utils

HOME = str(Path.home())

def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.prims.xform_prim_view import XFormPrimView
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.kit.viewport_legacy import get_viewport_interface
    from omni.isaac.synthetic_utils import SyntheticDataHelper
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

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()

    # Initialize Controller
    robot_interface = FmmIsaacInterface(robot_sim)
    lite_controller = FmmQPControl(
        dt=(1.0 / config["fps"]), fmm_mode="no-tower", robot=robot_interface.robot_model
    )
    # Initialize Camera
    my_camera = my_world.scene.add(
        XFormPrimView(
            prim_paths_expr="/World/franka_alt_fingers/panda_hand/geometry/realsense/realsense_camera",
            name="fancy_camera",
        )
    )

    print("CHECK camera: ", my_world.scene.object_exists(name="fancy_camera"))
    print(my_camera.get_world_poses()[0])
    viewport_interface = get_viewport_interface()
    viewport_handle = viewport_interface.create_instance()
    viewport_window = viewport_interface.get_viewport_window(viewport_handle)
    viewport_window.set_active_camera(
        "/World/FMM/wrist_camera/Camera"
    )
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
        my_world.reset()
        my_world.step(render=False)
        new_object_pos = initial_object_pos_selector()
        ee_initial_target_pose = gripper_inital_point_selector()
        my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
        grasp_pose = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)

        nav_flag = True
        pregrasp_flag = False
        grasp_flag = False
        pick_flag = False
        step_counter = 0
        step_threshold = 500
        while step_counter <= step_threshold:
            if nav_flag:
                init_dist, goal_reached, ee_twist = move_to_goal(
                    robot_interface, lite_controller, ee_initial_target_pose
                )
                robot_interface.update_robot_model()
                my_world.step(render=False)
                if init_dist < 0.02:
                    nav_flag = False
                    pregrasp_flag = True
                    print("Gripper in initial pose!")
                    gt = data_helper.get_groundtruth(
                        sensor_names=["rgb", "depth", "instanceSegmentation"],
                        viewport=viewport_window,
                    )
                    print("images collected from the environment!")
                    input_netwotk = input_creator(gt, fx, fy, x0, y0)
                    pred_grasps_cam, scores, contact_pts = network_inference(
                        in_paths=input_netwotk
                    )
                    scores = scores[-1]
                    max_idx = np.argmax(scores)
                    pred_points = pred_grasps_cam[-1][max_idx, :]
                    contact_points = contact_pts[-1][max_idx, :]
                    print("initial pred_grasps_cam: ", pred_points)
                    print("initial contact points: ", contact_points)
                    d = 10.34  # mentioned in the contact graspnet code (buid_6d_grasp function)
                    # ================ Alternative approach ==================
                    camera_pose = my_camera.get_world_poses()[0]
                    orientation_matrix = quater_rotation1(my_camera)
                    worldTcam = np.concatenate(
                        (
                            np.concatenate(
                                (orientation_matrix, camera_pose.reshape(3, 1)), axis=1
                            ),
                            np.array([0, 0, 0, 1]).reshape(1, 4),
                        )
                    )
                    camTnet = np.array(
                        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                    ).reshape(4, 4)
                    netTgrasp = pred_points
                    netTgrasp[:3, -1] *= 100
                    graspTgoal = np.eye(4)
                    graspTgoal[2, 3] = d
                    worldTgoal = worldTcam @ camTnet @ netTgrasp @ graspTgoal
                    final_picking_pose = worldTgoal[:3, -1]
                    # ================

                    R_g = pred_points[:3, :3]
                    t_g = pred_points[:3, -1]
                    b_vec = R_g[:, 0]
                    a_vec = R_g[:, -1]
                    picking_pose_rel = (t_g / 100) + a_vec * (d / 100)
                    cam_net_orientation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).reshape(
                        3, 3
                    )
                    print(
                        "c: ",
                        camera_pose
                        + (
                            100
                            * np.matmul(
                                np.matmul(orientation_matrix, cam_net_orientation),
                                contact_points,
                            )
                        ),
                    )
                    final_picking_pose2 = camera_pose + (
                        100
                        * np.matmul(
                            np.matmul(orientation_matrix, cam_net_orientation), picking_pose_rel
                        )
                    )

                    final_picking_pose = final_picking_pose.reshape(
                        3,
                    )
                    print("Final picking pose: ", final_picking_pose)
                    print("Final picking pose2: ", final_picking_pose2)
                    print("done")
            elif pregrasp_flag:
                pregrasp_dist, goal_reached, ee_twist = move_to_goal(robot_interface, lite_controller, grasp_pose)
                robot_interface.update_robot_model()
                my_world.step(render=False)
                if pregrasp_dist < 0.027:
                    pregrasp_flag = False
                    grasp_flag = True
            elif grasp_flag:
                distance_len, goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, grasp_pose)
                # print("before grasp_obj")
                obj_grasped = grasp_obj(robot_interface, lite_controller)
                # print("after grasp_obj")
                robot_interface.update_robot_model()
                my_world.step(render=False)
                if obj_grasped:
                    grasp_flag = False
                    pick_flag = True
            elif pick_flag:
                distance_len_f, goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, pick_pose)
                robot_interface.close_gripper()
                robot_interface.update_robot_model()
                my_world.step(render=False)
                if my_object.get_world_pose()[0][-1] > 0.98:
                    pick_flag = False
                    print("Object picked!")
                    break
                if distance_len_f < 0.03:
                    pick_flag = False
                    print("Failure!")
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
    obj_grasped = True if controller.grasp_counter > 40 else False
    return obj_grasped

def quater_rotation1(camera):
    quaternion = camera.get_world_poses()[1].reshape(
        4,
    )
    gf_quaternion = Gf.Quatf(
        float(quaternion[0]),
        Gf.Vec3f(float(quaternion[1]), float(quaternion[2]), float(quaternion[3])),
    )
    mat = Gf.Matrix3d(gf_quaternion).GetTranspose()
    return mat

if __name__ == "__main__":
     with open('simulation_config.yaml') as f:
         simulation_config = yaml.load(f, Loader=SafeLoader)
         print("simulation config: ", simulation_config)
     main(simulation_config)