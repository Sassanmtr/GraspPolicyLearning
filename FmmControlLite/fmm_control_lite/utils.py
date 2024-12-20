import numpy as np
from roboticstoolbox.robot.ERobot import ERobot



def link_collision_damper(
    e_robot: ERobot,
    shape,
    q=None,
    di=0.15,
    ds=0.01,
    xi=1.0,
    end=None,
    start=None,
    collision_list=None,
):
    """
    Formulates an inequality contraint which, when optimised for will
    make it impossible for the robot to run into a collision. Requires
    See examples/neo.py for use case
    :param ds: The minimum distance in which a joint is allowed to
        approach the collision object shape
    :type ds: float
    :param di: The influence distance in which the velocity
        damper becomes active
    :type di: float
    :param xi: The gain for the velocity damper
    :type xi: float
    :param from_link: The first link to consider, defaults to the base
        link
    :type from_link: Link
    :param to_link: The last link to consider, will consider all links
        between from_link and to_link in the robot, defaults to the
        end-effector link
    :type to_link: Link
    :returns: Ain, Bin as the inequality contraints for an omptimisor
    :rtype: ndarray(6), ndarray(6)
    """

    end, start, _ = e_robot._get_limit_links(start=start, end=end)

    links, n, _ = e_robot.get_path(start=start, end=end)
    _, start_idx, _ = e_robot.get_path(start=e_robot.base_link, end=start)
    start_idx -= 1

    j = 0
    Ain = None
    bin = None

    def indiv_calculation(link, link_col, q):
        # d: distance
        # wTlp: point on robot
        # wTcp: point on shape
        d, wTlp, wTcp = link_col.closest_point(shape, di)

        if d is not None:
            lpTcp = -wTlp + wTcp

            norm = lpTcp / d
            norm_h = np.expand_dims(np.concatenate((norm, [0, 0, 0])), axis=0)

            Je = e_robot.jacobe(q, start=e_robot.base_link, end=link, tool=link_col.T)
            n_dim = Je.shape[1]
            dp = norm_h @ shape.v
            l_Ain = np.zeros((1, start_idx+n))

            l_Ain[0, :n_dim] = norm_h @ Je
            l_bin = (xi * (d - ds) / (di - ds)) + dp
        else:
            l_Ain = None
            l_bin = None

        return l_Ain, l_bin

    for link in links:
        if link.isjoint:
            j += 1

        if collision_list is None:
            col_list = link.collision
        else:
            col_list = collision_list[j - 1]

        for link_col in col_list:
            l_Ain, l_bin = indiv_calculation(link, link_col, q)

            if l_Ain is not None and l_bin is not None:
                if Ain is None:
                    Ain = l_Ain
                else:
                    Ain = np.concatenate((Ain, l_Ain))

                if bin is None:
                    bin = np.array(l_bin)
                else:
                    bin = np.concatenate((bin, l_bin))

    return Ain, bin