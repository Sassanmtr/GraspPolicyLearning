from omni.isaac.kit import SimulationApp
from graspnet.contact_graspnet.inference import inference
import carb
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import spatialmath as sm
from spatialmath.base import trnorm

HOME = str(Path.home())
print("HOME: ", HOME)


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
            "target_position": sm.SE3.Rt(vertical_grasp, [4.3, -3, 0.5]),
        },
    }
    return observations


def input_creator(data, fx, fy, x0, y0):
    output = {}
    rgb = data["rgb"][:, :, 0:3]
    depth = data["depth"]
    K = np.array([fx, 0, x0, 0, fy, y0, 0, 0, 1]).reshape(3, 3)
    if "instanceSegmentation" in data.keys():
        instance = data["instanceSegmentation"][0]
        output["seg"] = instance
    output["rgb"] = rgb
    output["depth"] = depth
    output["K"] = K

    return output


def network_inference(in_paths):
    check_dir = "graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"

    pred_grasps_cam, scores, contact_pts = inference(
        checkpoint_dir=check_dir,
        input_paths=in_paths,
        z_range=eval(str([0.2, 1.8])),
        local_regions=False,
        filter_grasps=False,
        segmap_id=False,
        forward_passes=1,
    )
    print("Inference done!")
    return pred_grasps_cam, scores, contact_pts


def point_selector(
    lower=np.array([0.6, -0.1, 0.8]),
    upper=np.array([0.8, 0.1, 1]),
    y_rot=np.array([-33, -27]),
):
    x = np.random.uniform(low=lower[0], high=upper[0], size=1)
    y = np.random.uniform(low=lower[1], high=upper[1], size=1)
    z = np.random.uniform(low=lower[2], high=upper[2], size=1)
    theta = np.random.randint(low=y_rot[0], high=y_rot[1], size=1)
    r = R.from_euler("xyz", [180, theta[0], 0], degrees=True)
    target_ori = sm.SO3(np.array(r.as_matrix()))
    target_pos = [x[0], y[0], z[0]]
    target_pose = sm.SE3.Rt(target_ori, target_pos)
    return target_pose


def main(config):
    simulation_app = SimulationApp({"headless": False})
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.prims.xform_prim import XFormPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core import World
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

    # # Initialize Env
    # add_reference_to_stage(usd_path=config["env_path"], prim_path="/World/Env")
    # my_world.scene.add(XFormPrim(prim_path="/World/Env", name="env"))
    # print("env done!")

    # Add Robot
    asset_path = config["model_path"]
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/FMM")
    robot_sim = my_world.scene.add(
        Robot(prim_path="/World/FMM", name="fancy_fmm", position=[0, 0, 0])
    )
    print("CHECK FMM: ", my_world.scene.object_exists(name="fancy_franka"))
    print("robot done!")

    # my_rubik = my_world.scene.add(XFormPrim(prim_path="/World/Env/rubik", name="fancy_rubik"))
    # my_bleach = my_world.scene.add(
    #     XFormPrim(prim_path="/World/Env/bleach_cleanser", name="fancy_bleach")
    # )
    my_marker = my_world.scene.add(
        XFormPrim(prim_path="/World/Env/marker", name="fancy_marker")
    )
    # my_cracker = my_world.scene.add(XFormPrim(prim_path="/World/Env/cracker_box", name="fancy_cracker"))
    print("CHECK mesh: ", my_world.scene.object_exists(name="fancy_marker"))

    my_world.reset()
    robot_interface = FmmIsaacInterface(robot_sim)
    initial_controller = ReachLocation(robot_interface)
    target_pose = point_selector()
    final_controller = PickAndPlace(robot_interface)

    flag = True
    robot = robot_interface.robot_model
    counter = 0

    # Start simulation
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()
    while simulation_app.is_running():
        my_world.step(render=True)

        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
            observations = get_observations(robot=robot, obj=my_marker)
            if flag:
                initial_controller.move(target_pose)
                wTe = robot.fkine(robot.q)
                dist = np.linalg.norm(wTe.t - target_pose.t)
                if dist <= 0.012:
                    print("Gripper is in the initial grasping pose!")
                    gt = robot_interface.get_camera_data()
                    print("images collected from the environment!")


if __name__ == "__main__":
    config = {
        "env_path": "/home/mokhtars/Documents/isaac-codes/Scenes_Robots/Collected_franka_table/franka_table.usd",
        "model_path": HOME + "/Documents/isaac-fmm/models/fmm_full.usd",
        "fps": 50,
    }
    main(config)
