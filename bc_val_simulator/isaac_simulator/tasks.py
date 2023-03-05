from robot_control import PickAndPlace, ReachLocation
import spatialmath as sm
import numpy as np
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.physx.scripts import utils


class DemoControl:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.reset()
        return

    def reset(self):
        self.controller = ReachLocation(self.robot_interface)
        # Objects
        self.container = XFormPrim(
            prim_path="/World/Hospital/hospital/SM_Container_01a_17", name="container"
        )
        self.container.set_local_scale([0.5, 0.5, 0.5])
        self.container.set_local_pose([2.0, 0.5, 1.0])
        self.container.set_default_state([2.0, 0.5, 1.0])
        return

    def step(self):
        container_pos, _ = self.container.get_local_pose()
        desired_pos = container_pos + [0, 0, 0.2]
        vertical_grasp = sm.SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        grasp_pose = sm.SE3.Rt(vertical_grasp, desired_pos)
        self.controller.move(grasp_pose)
        return


class MoveContainer:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.reset()
        return

    def reset(self):
        self.controller = PickAndPlace(self.robot_interface)

        # Objects
        container_prim_path = "/World/Hospital/hospital/SM_Container_01a_17"
        self.container = GeometryPrim(
            prim_path=container_prim_path,
            name="container",
            collision=True,
        )
        utils.setRigidBody(self.container.prim, "convexHull", False)
        self.container.set_local_scale([0.5, 0.5, 0.5])
        self.container.set_local_pose([2.0, 0.5, 1.0])
        self.container.set_default_state([2.0, 0.5, 1.0])
        self.table = GeometryPrim(
            prim_path="/World/Hospital/hospital/SM_SideTable_02a2",
            name="table",
            collision=True,
        )
        return

    def get_objects_pose(self):
        container_pos, container_quat = self.container.get_local_pose()
        container_quat = sm.UnitQuaternion(
            container_quat
        ).vec  # needed to have proper norm
        container_mat = sm.SO3(sm.base.q2r(container_quat, order="sxyz"))
        table_pos, table_quat = self.table.get_local_pose()
        table_quat = sm.UnitQuaternion(table_quat).vec  # needed to have proper norm
        table_mat = sm.SO3(sm.base.q2r(table_quat, order="sxyz"))
        container_pose = sm.SE3.Rt(container_mat, container_pos)
        table_pose = sm.SE3.Rt(table_mat, table_pos)
        return container_pose, table_pose

    def step(self):
        container_pose, table_pose = self.get_objects_pose()
        # flat_grasp = sm.SO3(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))
        vertical_grasp = sm.SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        grasp_pose = sm.SE3.Rt(vertical_grasp, container_pose.t + [0, 0, 0.05])
        place_pose = sm.SE3.Rt(vertical_grasp, table_pose.t + [-0.4, 0, 1.0])
        self.controller.move(grasp_pose, place_pose)
        return


task_switch = {
    "DemoControl": DemoControl,
    "MoveContainer": MoveContainer,
}
