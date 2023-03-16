import sys
sys.path.append('/home/mokhtars/Documents/bc_network/bc_network')
from pathlib import Path
from omni.isaac.kit import SimulationApp
import carb
import numpy as np
from helpers import *
import torch
from bcnet import Policy

HOME = str(Path.home())
print("HOME: ", HOME)


def simulation_main(config, policy, lstm_state=None):
    simulation_app = SimulationApp({"headless":False})
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.physx.scripts import utils
    import omni.usd
    from isaac_simulator.fmm_isaac import FmmIsaacInterface
    from isaac_simulator.robot_control import FakePickAndPlace, ReachLocation

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
    fake_controller = FakePickAndPlace(robot_interface)

    navigation_flag = False
    pick_and_place_flag = True
    robot = robot_interface.robot_model

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    nr_traj = 200
    traj_counter = 0
    success_counter = 0
    failure_counter = 0
    step_counter = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
            observations = get_observations(robot=robot, obj=my_object)
            if (
                traj_counter == nr_traj
            ):  # Enough trajectories have been recorded

                break

            if (
                pick_and_place_flag
            ):  # One trajectory has been finished (successful or failrue) and a new one should start
                my_world.reset()
                new_object_pos = initial_object_pos_selector()
                gripper_target_pose = gripper_inital_point_selector()
                my_object.set_world_pose(np.array(new_object_pos), [1, 0, 0, 0])
                fake_controller.reset()
                navigation_flag = True
                pick_and_place_flag = False
                step_counter = 0
            if (
                navigation_flag
            ):  # Navigate the gripper to the initial pose before starting Pick and Place
                initial_controller.move(gripper_target_pose)
                wTe = robot.fkine(robot.q)
                dist = np.linalg.norm(wTe.t - gripper_target_pose.t)
                if (
                    dist <= 0.012
                ):  # If end effectors are close enough to the target initial pose, find the target pose for Pick and Place
                    wTgrasp = wTgrasp_finder(suc_grasps, robot, new_object_pos)
                    print("Final grasp: ", wTgrasp)
                    navigation_flag = False
            else:  # Pick and Place is performed
                current_ee_pose = observations[robot.name]["end_effector_position"]
                image_array, joint_array = predict_input_processing(robot_interface, robot_sim, device)
                next_action, lstm_state = policy.predict(image_array, joint_array, lstm_state)
                # if working with Euler use output_processing, if working with quat use q_output_processing
                print()
                print("Predicted quaternion norm: ", np.linalg.norm(next_action[3:-1]))
                target_pose = q_output_processing(current_ee_pose, next_action)
                initial_controller.move(target_pose)
                if next_action[-1] < 0:
                    robot_interface.close_gripper()
                else:
                    robot_interface.open_gripper()

                if fake_controller.done == True:  # success
                    target_dist = np.linalg.norm(
                        np.array(my_object.get_world_pose()[0])
                        - observations[my_object.name]["target_position"].t
                    )
                    print("Target_dist: ", target_dist)
                    if target_dist <= 0.25:
                        success_counter += 1
                        traj_counter += 1
                        print("Success!")
                    else:
                        failure_counter += 1
                        traj_counter += 1
                        print("Nope!")
                    pick_and_place_flag = True
                step_counter += 1
                print("step: ", step_counter)
                if step_counter == 200:  # failure
                    print("Failure!")
                    failure_counter += 1
                    traj_counter += 1
                    pick_and_place_flag = True

    simulation_app.close()
    return


if __name__ == "__main__":

    simulation_config = {
        "model_path": HOME + "/Documents/isaac-fmm/models/fmm_full.usd",
        "object_path": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/simple_bowl.usd",
        "fps": 50,
        "mesh_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/bowl/bowl.h5",
        "cube_dir": HOME
        + "/Documents/isaac-codes/Grasping_task/imitation_learning/cracker.usd",
    }
    network_config = {
        "visual_embedding_dim": 2394,
        "proprio_dim": 13,
        "action_dim": 8,
        "learning_rate": 1e-5,
        "weight_decay": 3e-4,
        "batch_size": 4,
        "sequence_len": 5,
        "num_epochs": 100,
        "buffer_capacity": 100,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(network_config, device)
    model_path = "saved_models/policy.pt"
    policy.load_state_dict(torch.load(model_path))

    simulation_main(simulation_config, policy)
    