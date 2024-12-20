import numpy as np
import swift
import time
import spatialmath as sm
import spatialgeometry as sg
from fmm_control_lite.fmm_model import FmmModel
from fmm_control_lite.fmm_control import FmmQPControl

robot_model = FmmModel()
env = swift.Swift()
env.launch()

ELBOW_LEFT_Q = np.array(
    [
        -1.4111348245688131,
        -1.2129180530427706,
        -0.13962625531258416,
        -2.358785666380864,
        0.46029318722089124,
        1.258500477120879,
        -0.5143043192269073,
    ]
)

Q_CURRENT = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        -1.26058975,
        -1.34609048,
        0.11537959,
        -2.49902826,
        0.50241926,
        1.60649943,
        0.13897193,
    ]
)

EE_START = sm.SE3(
    np.array(
        [
            [0.84132622, -0.00711519, 0.54048087, 0.46307772],
            [0.00374581, -0.9998126, -0.01899292, -0.13671148],
            [0.54051472, 0.01800378, -0.8411419, 0.979675],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    check=False,
)
while True:
    q_reference = np.concatenate((np.zeros(4), ELBOW_LEFT_Q))
    robot_model.q = Q_CURRENT
    fmm_control = FmmQPControl(dt=0.02, fmm_mode="arm-only", robot=robot_model)

    # goal_position = np.random.uniform([0.5, -0.5, 0.6], [1.0, 0.5, 0.8])
    target = sg.Sphere(radius=0.02, pose=EE_START)
    # wTeegoal = sm.SE3.Rt(R=sm.SO3.Rx(np.pi), t=goal_position)

    env.add(robot_model)
    env.add(target)
    for collision_shape in robot_model.tower_collision:
        env.add(collision_shape)

    distance = 10
    while distance > 0.04:
        qd, distance = fmm_control.wTeegoal_2_qd(EE_START, q_reference)

        # Apply action
        robot_model.qd = qd
        env.step(0.02)
    time.sleep(4)
    env.reset()

print("Done!")
