import numpy as np
from fmm_old_control.fmm import FmmModel
import omni
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils import prims
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.kit.viewport_legacy import get_viewport_interface
from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface


class FmmIsaacInterface:
    """
    robot_sim joint list:
      - 'dummy_x_joint',
      - 'dummy_y_joint',
      - 'dummy_z_joint',
      - 'ewellix_lift_top_joint',
      - 'panda_joint1',
      - 'panda_joint2',
      - 'panda_joint3',
      - 'panda_joint4',
      - 'panda_joint5',
      - 'panda_joint6',
      - 'panda_joint7',
      - 'panda_finger_joint1',
      - 'panda_finger_joint2'
    """

    def __init__(self, robot_sim):
        # Robotic Toolbox Model
        self.robot_model = FmmModel()
        self.robot_model.q = self.robot_model.qr

        # Isaac Robot initialization
        self.robot_sim = robot_sim
        self.robot_sim.set_joints_default_state(
            np.append(self.robot_model.qr, [0.05, 0.05])
        )
        self.robot_sim.set_joint_positions(np.append(self.robot_model.qr, [0.05, 0.05]))
        _, kds_gains = self.robot_sim.get_articulation_controller().get_gains()
        kds_gains[3] *= 10
        self.robot_sim.get_articulation_controller().set_gains(kds=kds_gains)

        # # Wrist Camera Initialization
        camera_prim = prims.create_prim(
            prim_path="/World/FMM/wrist_camera/Camera",
            prim_type="Camera",
            attributes={
                "focusDistance": 1,
                "focalLength": 24,
                "horizontalAperture": 20.955,
                "verticalAperture": 15.2908,
                "clippingRange": (0.01, 1000000),
                "clippingPlanes": np.array([1.0, 0.0, 1.0, 1.0]),
            },
        )
        # debug = camera.GetAttributes()
        viewport_interface = get_viewport_interface()
        gripper_vp_handle = viewport_interface.create_instance()
        gripper_vp_window = viewport_interface.get_viewport_window(gripper_vp_handle)
        gripper_vp_window.set_active_camera("/World/FMM/wrist_camera/Camera")
        gripper_vp_window.set_texture_resolution(500, 300)
        gripper_vp_window.set_window_pos(500, 500)
        gripper_vp_window.set_window_size(500, 300)
        self.gripper_vp_window = gripper_vp_window
        # Get existing viewport, set active camera
        perspective_vp_handle = viewport_interface.get_instance("Viewport")
        perspective_vp_window = viewport_interface.get_viewport_window(
            perspective_vp_handle
        )
        perspective_vp_window.set_active_camera("/OmniverseKit_Persp")
        perspective_vp_window.set_camera_position(
            "/OmniverseKit_Persp", 2.5, 2.5, 2.5, True
        )
        perspective_vp_window.set_camera_target(
            "/OmniverseKit_Persp", 0.0, 0.0, 0.0, True
        )
        self.perspective_vp_window = perspective_vp_window

        self.data_helper = SyntheticDataHelper()

        # Lidars Initialization
        self.front_lidar_path = "/World/FMM/front_lidar/Front_Lidar"
        self.rear_lidar_path = "/World/FMM/rear_lidar/Rear_Lidar"
        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=self.front_lidar_path,
            parent=None,
            min_range=0.2,
            max_range=10.0,
            draw_points=True,
            draw_lines=False,
            horizontal_fov=270.0,
            vertical_fov=0.0,
            horizontal_resolution=1.0,
            vertical_resolution=1.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )
        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=self.rear_lidar_path,
            parent=None,
            min_range=0.2,
            max_range=10.0,
            draw_points=True,
            draw_lines=False,
            horizontal_fov=270.0,
            vertical_fov=0.0,
            horizontal_resolution=1.0,
            vertical_resolution=1.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )
        self.lidars_interface = acquire_lidar_sensor_interface()

        # Model analysis
        print(self.robot_sim.dof_names)
        return

    def get_lidar_data(self):
        front_pc = self.lidars_interface.get_point_cloud_data(self.front_lidar_path)
        rear_pc = self.lidars_interface.get_point_cloud_data(self.rear_lidar_path)
        return front_pc, rear_pc

    def get_camera_data(self):
        gt = self.data_helper.get_groundtruth(
            sensor_names=["rgb", "depth"], viewport=self.gripper_vp_window
        )
        return gt

    def update_robot_model(self):
        q = self.robot_sim.get_joint_positions()[:-2]
        qd = self.robot_sim.get_joint_velocities()[:-2]
        self.robot_model.q = q
        self.robot_model.qd = qd
        return

    def move_joints(self, qd):
        action = ArticulationAction(
            joint_velocities=qd,
            joint_indices=list(range(self.robot_model.n)),
        )
        self.robot_sim.apply_action(action)
        return

    def close_gripper(self):
        action = ArticulationAction(
            joint_efforts=[-70, -70],
            joint_indices=[11, 12],
        )
        self.robot_sim.apply_action(action)
        return

    def open_gripper(self):
        action = ArticulationAction(
            joint_efforts=[70, 70],
            joint_indices=[11, 12],
        )
        self.robot_sim.apply_action(action)
        return
