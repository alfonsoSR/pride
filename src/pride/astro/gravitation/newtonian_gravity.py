import numpy as np


def calculate_newtonian_potential(
    massive_bodies_gm: np.ndarray,
    relative_distances: np.ndarray,
) -> np.ndarray:
    """Newtonian potential from GMs and relative distances

    The Newtonian gravitational potential at distance r from a point mass M is given by: U = GM/r. This function calculates the total potential at a target point due to multiple massive bodies by summing their individual potentials.

    In the description of the arguments, M is the number of massive bodies, N is the number of points at which the potential is calculated, and the tuples indicate the dimensions of the arrays.

    :param massive_bodies_gm: Gravitational parameters of the massive bodies (M,)
    :param relative_distances: Relative distances from the target point to the massive bodies (N,)
    :return: Total Newtonian potential (N,)
    """

    return np.sum(massive_bodies_gm[:, None] / relative_distances, axis=0)


def calculate_newtonian_potential_from_bcrf_positions(
    massive_bodies_gm: np.ndarray,
    x_target_bcrf: np.ndarray,
    x_bodies_bcrf: np.ndarray,
) -> np.ndarray:
    """Newtonian potential from GMs and BCRF positions

    The Newtonian gravitational potential at distance r from a point mass M is given by: U = GM/r. This function calculates the total potential at a target point due to multiple massive bodies by summing their individual potentials.

    In the description of the arguments, M is the number of massive bodies, N is the number of points at which the potential is calculated, and the tuples indicate the dimensions of the arrays.

    :param massive_bodies_gm: Gravitational parameters of the massive bodies (M,)
    :param x_target_bcrf: BCRF position of the target point (N, 3)
    :param x_bodies_bcrf: BCRF positions of the massive bodies (M, N, 3)
    :return: Total Newtonian potential (N,)
    """

    # Relative position of target with respect to massive bodies
    x_target_bodies = x_target_bcrf[None, :, :] - x_bodies_bcrf
    r_target_bodies = np.linalg.norm(x_target_bodies, axis=-1)

    # Calculate Newtonian potential
    return calculate_newtonian_potential(massive_bodies_gm, r_target_bodies)
