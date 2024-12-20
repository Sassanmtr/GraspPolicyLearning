import numpy as np
import spatialmath as sm
import qpsolvers as qp

from fmm_control_lite.utils import link_collision_damper

DEFAULT_WEIGHTS = {
    "base_qd": 0.0000001,
    "tower_qd": 1.0,
    "arm_qd": 0.01,
    "slack": 2.0,
    "q_des": 1.0,
}


class FmmQPControl:
    def __init__(
        self, dt, fmm_mode, robot, cost_w=DEFAULT_WEIGHTS, tower_avoidance=False
    ) -> None:
        self.dt = dt
        self.blocked_joints = self.get_blocked_joints(fmm_mode)
        self.robot = robot
        self.cost_w = cost_w
        self.tower_avoidance = tower_avoidance
        self.last_qd_main = None
        self.last_qd_null = None
        self.last_qd_cmd = None
        self.grasp_counter = 0

        # Workaround for collision bug. TODO: Investigate proper fix
        import swift

        env = swift.Swift()
        env.launch(headless=True)
        env.add(self.robot)
        return

    def get_blocked_joints(self, mode):
        if mode == "full":
            blocked_joints = []
        elif mode == "no-tower":
            blocked_joints = [3]
        elif mode == "no-base":
            blocked_joints = [0, 1, 2]
        elif mode == "arm-only":
            blocked_joints = [0, 1, 2, 3]
        elif mode == "base-only":
            blocked_joints = [3, 4, 5, 6, 7, 8, 9, 10]
        else:
            raise NotImplementedError("We do not have this mode!")
        return blocked_joints

    def wTeegoal_2_qd(self, wTeegoal, q_reference=None):
        wTee = self.robot.fkine(self.robot.q)
        eeTeegoal = wTee.inv() * wTeegoal
        qd, distance_lin, distance_ang = self.eeTeegoal_2_qd(eeTeegoal, q_reference)
        return qd, distance_lin, distance_ang

    def eeTeegoal_2_qd(self, eeTeegoal, q_reference=None):
        ee_twist, error_t, error_rpy = self.eeTeegoal_2_eetwist(eeTeegoal)
        qd = self.ee_twist_2_qd(ee_twist, q_reference)
        distance_lin = np.linalg.norm(error_t)
        distance_ang = np.linalg.norm(error_rpy)
        return qd, distance_lin, distance_ang

    def wTeegoal_2_eetwist(self, wTeegoal):
        wTee = self.robot.fkine(self.robot.q)
        eeTeegoal = wTee.inv() * wTeegoal
        ee_twist, error_t, error_rpy = self.eeTeegoal_2_eetwist(eeTeegoal)
        return ee_twist, error_t, error_rpy

    def eeTeegoal_2_eetwist(self, eeTeegoal):
        error_t = eeTeegoal.t
        error_rpy = sm.smb.tr2rpy(eeTeegoal.R, unit="rad", order="zyx", check=False)
        v = error_t / self.robot.qdd_tau  # to have time to slow down
        w = error_rpy / self.robot.qdd_tau  # to have time to slow down
        ee_twist = np.concatenate((v, w))
        ee_twist = np.clip(
            ee_twist, a_min=self.robot.xd_lb, a_max=self.robot.xd_ub
        )
        return ee_twist, error_t, error_rpy

    def get_ineq_constraints(self, n, x_size):
        ## L1 loss
        A_in = np.r_[
            np.c_[np.eye(n), -np.eye(n), np.zeros((n, x_size - (2 * n)))],
            np.c_[-np.eye(n), -np.eye(n), np.zeros((n, x_size - (2 * n)))],
        ]
        b_in = np.zeros(n * 2)

        ## Tower Obstacle Avoidance
        if self.tower_avoidance:
            for collision_shape in self.robot.tower_collision:
                c_Ain, c_bin = link_collision_damper(
                    self.robot,
                    shape=collision_shape,
                    q=self.robot.q,
                    start=self.robot.link_dict["panda_link3"],
                    end=self.robot.link_dict["panda_link7"],
                )

                if c_Ain is not None and c_bin is not None:
                    c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], x_size - n))]

                    # Stack the inequality constraints
                    A_in = np.r_[A_in, c_Ain]
                    b_in = np.r_[b_in, c_bin]
        return A_in, b_in

    def get_lower_upper_bounds(self, n, q, last_qd, x_size, blocked_joints):
        # to have time to slow down
        q_lb_delta = (self.robot.q_lb - q) / self.robot.qdd_tau
        q_ub_delta = (self.robot.q_ub - q) / self.robot.qdd_tau

        qdd_lb_delta = self.robot.qdd_lb * self.dt + last_qd[:n]
        qdd_ub_delta = self.robot.qdd_ub * self.dt + last_qd[:n]

        qd_lb = np.fmax.reduce([q_lb_delta, self.robot.qd_lb, qdd_lb_delta])
        qd_ub = np.fmin.reduce([q_ub_delta, self.robot.qd_ub, qdd_ub_delta])

        lb = np.r_[qd_lb, -1000 * np.ones(x_size - n)]
        ub = np.r_[qd_ub, 1000 * np.ones(x_size - n)]

        # Joints Masking
        if blocked_joints is None:
            blocked_joints = self.blocked_joints
        for idx in blocked_joints:
            lb[idx] = 0.0
            ub[idx] = 0.0
        return lb, ub

    def solve_qp(self, Q, c, A_eq, b_eq, A_in, b_in, lb, ub, last_qd, x_size):
        qd = qp.solve_qp(
            Q,
            c,
            A=A_eq,
            b=b_eq,
            G=A_in,
            h=b_in,
            lb=lb,
            ub=ub,
            initvals=last_qd,
            solver="osqp",
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=False,
        )
        if qd is None:
            print("Main Optimization Failed!")
            qd = np.zeros(x_size)
        return qd

    def ee_twist_qp(self, ee_twist_des, last_qd, q=None, blocked_joints=None):
        # Variable Definition: x = [q, t_l1, slack]
        n = self.robot.n
        x_size = 2 * n + 6
        if last_qd is None:
            last_qd = np.zeros(x_size)
        if q is None:
            q = self.robot.q

        # Q-Matrix
        Q = np.zeros((x_size, x_size))
        Q[n * 2 :, n * 2 :] = self.cost_w["slack"] * np.eye(6)  # slack

        # c-vector
        c = np.zeros(x_size)
        ## l1 loss weight
        c[n : n * 2] = (
            [self.cost_w["base_qd"]] * 3
            + [self.cost_w["tower_qd"]]
            + [self.cost_w["arm_qd"]] * 7
        )

        # Equality contraints -> the main objective (J*q + slack = v_des)
        A_eq = np.c_[self.robot.jacobe(self.robot.q), np.zeros((6, n)), np.eye(6)]
        b_eq = ee_twist_des

        # Inequality constraints
        A_in, b_in = self.get_ineq_constraints(n, x_size)

        # The lower and upper bounds on the joint velocity and slack variable
        lb, ub = self.get_lower_upper_bounds(n, q, last_qd, x_size, blocked_joints)

        # Solve for the joint velocities dq
        qd = self.solve_qp(Q, c, A_eq, b_eq, A_in, b_in, lb, ub, last_qd, x_size)
        return qd

    def q_des_qp(self, goal_q, last_qd, q=None, blocked_joints=None):
        # Variable Definition: x = [q, t_l1]
        # ghost_q is the one you want to move
        # state_q is the one you want to reach

        n = self.robot.n
        x_size = 2 * n
        if last_qd is None:
            last_qd = np.zeros(x_size)
        if q is None:
            q = self.robot.q

        # Q-Matrix
        Q = np.zeros((x_size, x_size))
        Q[3:n, 3:n] = self.cost_w["q_des"] * np.eye(n - 3)

        # c-vector
        c = np.zeros(x_size)
        ## qready cost
        dist_to_qready = (q - goal_q)[3:n]
        c[3:n] = dist_to_qready @ (self.cost_w["q_des"] * np.eye(n - 3))
        ## l1 loss weight
        c[n : n * 2] = (
            [self.cost_w["base_qd"]] * 3
            + [self.cost_w["tower_qd"]]
            + [self.cost_w["arm_qd"]] * 7
        )

        # Equality contraints
        A_eq = None
        b_eq = None

        # Inequality constraints
        A_in, b_in = self.get_ineq_constraints(n, x_size)

        # The lower and upper bounds on the joint velocity and slack variable
        lb, ub = self.get_lower_upper_bounds(n, q, last_qd, x_size, blocked_joints)

        # Solve for the joint velocities dq
        qd = self.solve_qp(Q, c, A_eq, b_eq, A_in, b_in, lb, ub, last_qd, x_size)
        return qd

    def ee_twist_2_qd(self, ee_twist_des, q_reference=None):
        if q_reference is None:
            q_reference = self.robot.q_ready
        # TODO: Make the two qp run in parallel
        # Clip input
        ee_twist_des = np.clip(
            ee_twist_des, a_min=self.robot.xd_lb, a_max=self.robot.xd_ub
        )
        # Get main qd
        qd_main = self.ee_twist_qp(ee_twist_des, self.last_qd_main)
        self.last_qd_main = qd_main
        qd_main = qd_main[: self.robot.n]
        # Get null qd
        qd_null = self.q_des_qp(q_reference, self.last_qd_null)
        self.last_qd_null = qd_null
        qd_null = qd_null[: self.robot.n]
        # Get nullspace projection
        Je = self.robot.jacobe(self.robot.q)
        for idx in self.blocked_joints:
            Je[:, idx] = np.zeros_like(Je[:, idx])  # Mask unwanted joints
        N_proj = np.eye(self.robot.n) - (np.linalg.pinv(Je) @ Je)
        # Compute final qd
        qd = qd_main + (N_proj @ qd_null)
        self.last_qd_cmd = qd
        return qd

    def handguide_2_qd(self, q_des, q_reference=None):
        if q_reference is None:
            q_reference = self.robot.q_ready
        # TODO: Make the two qp run in parallel
        # Clip input
        q_des = np.clip(q_des, a_min=self.robot.q_lb, a_max=self.robot.q_ub)
        # Get main qd
        qd_main = self.q_des_qp(q_des, self.last_qd_main, blocked_joints=[0, 1, 2, 3])
        self.last_qd_main = qd_main
        qd_main = qd_main[: self.robot.n]
        # Get null qd
        qd_null = self.q_des_qp(q_reference, self.last_qd_null)
        self.last_qd_null = qd_null
        qd_null = qd_null[: self.robot.n]
        # Get nullspace projection
        Je = self.robot.jacobe(self.robot.q)
        for idx in self.blocked_joints:
            Je[:, idx] = np.zeros_like(Je[:, idx])  # Mask unwanted joints
        N_proj = np.eye(self.robot.n) - (np.linalg.pinv(Je) @ Je)
        # Compute final qd
        qd = qd_main + (N_proj @ qd_null)
        self.last_qd_cmd = qd
        return qd
