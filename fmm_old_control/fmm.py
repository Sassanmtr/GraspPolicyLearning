import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from spatialmath import SE3
from pathlib import Path


class FmmModel(ERobot):
    """
    Class that imports our FMM robot URDF model

    ``Fmm()`` is a class which imports a Fmm robot definition
    from a URDF file. The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Fmm()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration
    """

    def __init__(self):
        home_path = str(Path.home())
        links, _, urdf_string, urdf_filepath = self.URDF_read(
            file_path="fmm_full.urdf", tld=home_path + "/Documents/isaac-fmm/models/"
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

        # Constraints
        self.q_ub = self.qlim[1]
        self.q_lb = self.qlim[0]
        self.qd_ub = np.array(
            [
                0.5,
                0.5,
                0.5,
                0.1,
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

        self.qr = np.array([0, 0, 0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(11)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
        print(self)
        return


if __name__ == "__main__":  # pragma nocover
    pass
