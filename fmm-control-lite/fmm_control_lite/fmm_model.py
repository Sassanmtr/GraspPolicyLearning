import pathlib
import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from spatialmath import SE3


class FmmModel(ERobot):
    """
    Class that imports our FMM robot URDF model.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Fmm()
        >>> print(robot)

    Defined configurations are:

    - q_zero    ->  zero joint angle configuration, 'L' shaped configuration
    - q_ready   ->  vertical 'READY' configuration
    - q_ub      ->  the joint positions upper bound
    - q_lb      ->  the joint positions lower bound
    - qd_ub     ->  the joint velocities upper bound
    - qd_lb     ->  the joint velocities lower bound
    """

    def __init__(self):
        urdf_path = pathlib.Path(__file__).parent.parent / "urdf/fmm_full.urdf"
        links, _, urdf_string, urdf_filepath = self.URDF_read(
            file_path=urdf_path.name, tld=urdf_path.parent
        )
        self.arm_base_link = [link for link in links if link.name == "panda_link0"][0]
        self.gripper_link = [link for link in links if link.name == "panda_hand"][0]

        super().__init__(
            links,
            name="Fmm",
            manufacturer="Custom",
            gripper_links=self.gripper_link,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        self.grippers[0].tool = SE3(0, 0, 0.1034)

        # Cartesian Space Constraints
        self.xd_ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.xd_lb = -self.xd_ub

        # Joint Space Constraints
        self.q_ub = self.qlim[1]
        self.q_lb = self.qlim[0]
        self.q_ub[9:] *= 0.95  # safety limit on the last franka joints
        self.q_lb[9:] *= 0.95  # safety limit on the last franka joints
        self.qd_ub = np.array(
            [
                0.2,
                0.2,
                0.2,
                0.2,
                2.1750,
                2.1750,
                2.1750,
                2.1750,
                2.6100,
                2.6100,
                2.6100,
            ]
        )
        self.qd_lb = -self.qd_ub
        # Time constant, i.e. how much time to reach velocity limits [seconds]
        self.qdd_tau = 0.05
        self.qdd_ub = self.qd_ub / self.qdd_tau
        self.qdd_lb = self.qd_lb / self.qdd_tau

        self.q_zero = np.zeros(11)
        self.q_ready = np.array([0, 0, 0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        self.qr = np.array([0, 0, 0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(11)

        self.addconfiguration("q_zero", self.q_zero)
        self.addconfiguration("q_ready", self.q_ready)
        self.addconfiguration("q_lb", self.q_lb)
        self.addconfiguration("q_ub", self.q_ub)
        print(self)
        return
