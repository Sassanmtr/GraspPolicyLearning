import sys
sys.path.append('/home/mokhtars/Documents/bc_network')
from pathlib import Path
import pathlib
import time
import yaml
import torch
from yaml.loader import SafeLoader
import numpy as np
import spatialmath as sm
from omni.isaac.kit import SimulationApp
from utils.helpers import *
from bc_network.bcnet import Policy_twist

HOME = str(Path.home())
print("HOME: ", HOME)

def main(config, policy, lstm_state=None):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
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
    
    step_counter = 0
    traj_counter = 0

    # my_world.set_simulation_dt(physics_dt=1.0 / 40.0, rendering_dt=1.0 / 10000.0)
    
    while simulation_app.is_running():
        # Move to pose 1
        while traj_counter < 10:
            my_world.reset()
            robot_interface.update_robot_model()
            my_world.step(render=True)
            new_object_pos = initial_object_pos_selector()
            pose1 = gripper_inital_point_selector()
            my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
            
            init_dist = 1
            # pose1 = gripper_target_pose
            
            while init_dist > 0.03:
                print("my_world.current_time_step_index", my_world.current_time_step_index)
                init_dist, goal_reached, ee_twist = move_to_goal(
                    robot_interface, lite_controller, pose1
                )
                robot_interface.update_robot_model()
                my_world.step(render=True)
            print("Reached goal")
            if traj_counter == 0:
                for i in range(2):
                    gt = robot_interface.get_camera_data()
                    print("gt: ", gt["rgb"].shape)
                    my_world.step(render=True)
            grasp_pose = wTgrasp_finder(suc_grasps, robot_interface.robot_model, new_object_pos)
            print("grasp_pose: ", grasp_pose)
            while step_counter < 300:
                image_array, joint_array = predict_input_processing(robot_interface, robot_sim, device)
                # print("my_world.current_time_step_index", my_world.current_time_step_index)
                next_action, lstm_state = policy.predict(
                    image_array, joint_array, lstm_state)
                print("network output: ", next_action[:-1])
                ee_twist, error_t, error_rpy = lite_controller.wTeegoal_2_eetwist(grasp_pose)
                print("controller output: ", ee_twist)
                print("---------------------------------")
                # qd = lite_controller.ee_twist_2_qd(next_action[:-1])
                qd = lite_controller.ee_twist_2_qd(ee_twist)
                robot_interface.move_joints(qd)
                robot_interface.update_robot_model()
                my_world.step(render=True)
                step_counter += 1
                # print("step_counter", step_counter)
            step_counter = 0
            traj_counter += 1
            
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



if __name__ == "__main__":

    with open('simulation_config.yaml') as f:
        simulation_config = yaml.load(f, Loader=SafeLoader)
        print("simulation config: ", simulation_config)
    with open('config.yaml') as f:
        network_config = yaml.load(f, Loader=SafeLoader)
        print("network config: ", network_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy_twist(network_config, device)
    model_path = "saved_models/policy.pt"
    policy.load_state_dict(torch.load(model_path))

    main(simulation_config, policy)
