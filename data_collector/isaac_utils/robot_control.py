import numpy as np
import qpsolvers as qp
import spatialmath as sm
import roboticstoolbox as rtb
from isaac_utils.pointcloud_processing import get_obstacle_map


def ee_pose_control(robot_interface, wTeg):
    """
    Controls the robot to reach the desired end-effector goal pose
    robot_interface: instance of a robot interface
    wTeg: the desired end-effector goal pose, in world frame (SE3)
    """
    robot = robot_interface.robot_model
    # Tuning parameters
    lambda_w = 0.1

    # Distance to goal
    wTe = robot.fkine(robot.q)
    eTeg = sm.SE3(np.linalg.inv(wTe.A) @ wTeg.A)
    distance = np.linalg.norm(eTeg.t)

    # Quadratic component of objective function
    Q = np.eye(robot.n + 6)

    # Joint velocity component of Q
    Q[: robot.n, : robot.n] *= lambda_w
    Q[3, 3] *= 100  # tower
    Q[:3, :3] *= 1.0 / distance  # base

    # Slack component of Q
    Q[robot.n :, robot.n :] = (1.0 / distance) * np.eye(6)

    # ee_twist
    v, _ = rtb.p_servo(wTe, wTeg, 1.5)
    v[3:] *= 1.3

    # The equality contraints
    Aeq = np.c_[robot.jacobe(robot.q), np.eye(6)]
    beq = v.reshape((6,))

    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (
            np.zeros(4),
            -robot.jacobm(start=robot.arm_base_link).reshape((robot.n - 4,)),
            np.zeros(6),
        )
    )

    # Get base to face end-effector
    theta_w = 0.1
    wTb = robot.fkine(robot.q, end=robot.links[3])
    bTe = sm.SE3(np.linalg.inv(wTb.A) @ wTe.A)
    theta = np.arctan2(bTe.t[1], bTe.t[0])
    epsilon = theta_w * theta
    c[2] += -epsilon

    # The lower and upper bounds on the joint velocity and slack variable
    q_lb = robot.q_lb - robot.q  # assumed dt = 1
    q_ub = robot.q_ub - robot.q  # assumed dt = 1
    qd_lb = np.fmax(q_lb, robot.qd_lb)
    qd_ub = np.fmin(q_ub, robot.qd_ub)
    lb = np.r_[qd_lb, -1000 * np.ones(6)]
    ub = np.r_[qd_ub, 1000 * np.ones(6)]

    # Obstacle Constraints
    pc, d = get_obstacle_map(robot_interface)
    n_obst = pc.shape[0]
    if n_obst == 0:
        Ain = np.zeros(robot.n + 6)
        bin = np.zeros(1)
    else:
        pc_rotated = wTb.R[:2, :2] @ pc.T
        base_r = 0.62
        Ain = np.concatenate((pc_rotated.T, np.zeros((n_obst, robot.n + 4))), axis=1)
        bin = d * (d - base_r)

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, A=Aeq, b=beq, G=Ain, h=bin, lb=lb, ub=ub, solver="osqp")
    if qd is None:
        qd = np.zeros(robot.n)
        print("Optimization Failed!")
    qd = qd[: robot.n]

    # Return "goal reached" flag and joint velocities
    if distance < 0.02:   # the original is 0.02, but I used 0.04
        return True, qd
    return False, qd


class ReachLocation:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        return

    def move(self, goal_pose):
        self.robot_interface.update_robot_model()
        self.grasp_pose_reached, qd = ee_pose_control(self.robot_interface, goal_pose)
        self.robot_interface.move_joints(qd)
        return


class PickAndPlace:
    def __init__(self, robot_interface):
        self.pregrasp_pose_reached = False
        self.grasp_pose_reached = False
        self.obj_grasped = False
        self.postgrasp_pose_reached = False
        self.preplace_pose_reached = False
        self.place_pose_reached = False
        self.obj_placed = False
        self.grasp_counter = 0
        self.place_counter = 0
        self.robot_interface = robot_interface
        self.last_pregrasp_pose = None
        self.last_preplace_pose = None
        self.save_mode = True
        self.done = False
        return

    def move(self, grasp_pose, place_pose):
        # self.move_to_pregrasp(grasp_pose)
        if not self.pregrasp_pose_reached:
            self.pregrasp_pose_reached = self.move_to_pregrasp(grasp_pose)
            print("A!")
        elif not self.grasp_pose_reached:
            self.grasp_pose_reached = self.move_to_grasp(grasp_pose)
            print("B!")
        elif not self.obj_grasped:
            _ = self.move_to_grasp(grasp_pose)
            self.obj_grasped = self.grasp_obj()
            print("C!")
        elif not self.postgrasp_pose_reached:
            self.postgrasp_pose_reached = self.move_to_postgrasp()
            print("D!")
        elif not self.preplace_pose_reached:
            self.preplace_pose_reached = self.move_to_preplace(place_pose)
            self.save_mode = False
            self.done = True
            print("E!")
        elif not self.place_pose_reached:
            self.place_pose_reached = self.move_to_place(place_pose)
            print("F!")
        elif not self.obj_placed:
            _ = self.move_to_place(place_pose)
            self.obj_placed = self.place_obj()
            print("G!")
            
        else:
            _ = self.move_to_postplace()
            print("Cube Moved!")

    def move_to_pregrasp(self, grasp_pose):
        grTpregr = sm.SE3.Trans(0.0, 0.0, -0.05)
        wTgr = grasp_pose
        wTpregr = sm.SE3(wTgr.A @ grTpregr.A)
        self.last_pregrasp_pose = wTpregr
        self.robot_interface.update_robot_model()
        reached_flag, qd = ee_pose_control(self.robot_interface, wTpregr)
        qd = qd / 10
        self.robot_interface.move_joints(qd)
        return reached_flag

    def move_to_grasp(self, grasp_pose):
        self.robot_interface.update_robot_model()
        reached_flag, qd = ee_pose_control(self.robot_interface, grasp_pose)
        qd = qd / 4
        self.robot_interface.move_joints(qd)
        return reached_flag

    def grasp_obj(self):
        self.robot_interface.close_gripper()
        self.grasp_counter += 1
        obj_grasped = True if self.grasp_counter > 20 else False
        return obj_grasped

    def move_to_postgrasp(self):
        self.robot_interface.update_robot_model()
        desired_pos = self.last_pregrasp_pose.t + [0, 0, 0.2]
        vertical_grasp = sm.SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        postgrasp_pose = sm.SE3.Rt(vertical_grasp, desired_pos)
        reached_flag, qd = ee_pose_control(self.robot_interface, postgrasp_pose)
        self.robot_interface.move_joints(qd)
        return reached_flag

    def move_to_preplace(self, place_pose):
        grTprepl = sm.SE3.Trans(0.0, 0.0, -0.05)
        wTgr = place_pose
        wTprepl = sm.SE3(wTgr.A @ grTprepl.A)
        self.last_preplace_pose = wTprepl
        self.robot_interface.update_robot_model()
        reached_flag, qd = ee_pose_control(self.robot_interface, wTprepl)
        self.robot_interface.move_joints(qd)
        return reached_flag

    def move_to_place(self, place_pose):
        self.robot_interface.update_robot_model()
        reached_flag, qd = ee_pose_control(self.robot_interface, place_pose)
        self.robot_interface.move_joints(qd)
        return reached_flag

    def place_obj(self):
        self.robot_interface.open_gripper()
        self.place_counter += 1
        obj_placed = True if self.place_counter > 100 else False
        return obj_placed

    def move_to_postplace(self):
        self.robot_interface.update_robot_model()
        reached_flag, qd = ee_pose_control(
            self.robot_interface, self.last_preplace_pose
        )
        self.robot_interface.move_joints(qd)
        return reached_flag

    def reset(self):
        self.pregrasp_pose_reached = False
        self.grasp_pose_reached = False
        self.obj_grasped = False
        self.postgrasp_pose_reached = False
        self.preplace_pose_reached = False
        self.place_pose_reached = False
        self.obj_placed = False
        self.grasp_counter = 0
        self.place_counter = 0
        self.last_pregrasp_pose = None
        self.last_preplace_pose = None
        self.done = False
        self.save_mode = True
        return
