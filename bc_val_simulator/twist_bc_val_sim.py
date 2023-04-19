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
from bc_network.bcnet import Policy

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
    init_dist = 1
    final_dist = 1
    step_counter = 0
    success_counter = 0
    # # Create directory to save data
    # traj_path = os.path.join(
    #     os.getcwd() + "/collected_data",
    #     "traj{}".format(success_counter),
    # )
    # os.mkdir(traj_path)
    # rgb_path = os.path.join(
    #     os.getcwd() + "/collected_data",
    #     "traj{}/rgb".format(success_counter),
    # )
    # os.mkdir(rgb_path)
    # depth_path = os.path.join(
    #     os.getcwd() + "/collected_data",
    #     "traj{}/depth".format(success_counter),
    # )
    # os.mkdir(depth_path)
    # pose_dict_name = "pose{}".format(success_counter)
    # pose_dict_name = {}
    my_world.set_simulation_dt(physics_dt=1.0 / 40.0, rendering_dt=1.0 / 10000.0)
    while simulation_app.is_running():
        # Move to pose 1
        new_object_pos = initial_object_pos_selector()
        gripper_target_pose = gripper_inital_point_selector()
        my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
        pose1 = gripper_target_pose
        while init_dist > 0.01:
            # print("my_world.current_time_step_index", my_world.current_time_step_index)
            init_dist, goal_reached, ee_twist = move_to_goal(
                robot_interface, lite_controller, pose1
            )
            robot_interface.update_robot_model()
            my_world.step(render=False)
            # time.sleep(1.0)
        print("Reached goal")
        # Move to pose 2
        pose2 = sm.SE3.Tx(0.5) * pose1
        print("Final grasp: ", pose2)
        
        while step_counter < 440:
            if step_counter == 0:
                # Save RGB and Depth images
                print("my_world.current_time_step_index", my_world.current_time_step_index)
                current_data = robot_interface.get_camera_data()
                rgb_image = cv2.cvtColor(current_data["rgb"][:, :, 0:3], cv2.COLOR_BGR2RGB)
                cv2.imwrite("collected_data/traj{}/rgb/{}.png".format(success_counter,
                        step_counter), rgb_image)
                cv2.imwrite("collected_data/traj{}/depth/{}.png".format(success_counter,
                        step_counter), current_data["depth"])
            
            image_array, joint_array = predict_input_processing(robot_interface, robot_sim, device)
            next_action, lstm_state = policy.predict(image_array, joint_array, lstm_state)
            qd = lite_controller.ee_twist_2_qd(next_action[:-1])
            robot_interface.move_joints(qd)
            robot_interface.update_robot_model()
            my_world.step(render=False)
            step_counter += 1
            print("step_counter", step_counter)
     
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



if __name__ == "__main__":

    with open('simulation_config.yaml') as f:
        simulation_config = yaml.load(f, Loader=SafeLoader)
        print("simulation config: ", simulation_config)
    with open('config.yaml') as f:
        network_config = yaml.load(f, Loader=SafeLoader)
        print("network config: ", network_config)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(network_config, device)
    model_path = "saved_models/policy.pt"
    policy.load_state_dict(torch.load(model_path))

    main(simulation_config, policy)
