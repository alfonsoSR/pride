import numpy as np
from ...constants import CLIGHT


def post_newtonian_near_field_effect(
    bodies_gm: np.ndarray,
    x_obs_bcrf_rx: np.ndarray,
    x_src_bcrf_tx: np.ndarray,
    x_bodies_bcrf_rx: np.ndarray,
    x_bodies_bcrf_tx: np.ndarray,
    consider_bending: bool = True,
) -> np.ndarray:
    """Post-Newtonian gravitational effect due to moving solar system bodies

    This function calculates a relativistic perturbation for the light-time equation of a signal travelling from a near-field source and an observer. The perturbation accounts for the reduction in the coordinate velocity of the signal, and the bending of the light path due to gravitational interactions with the massive bodies in the solar system.

    The algorithm is described in section 8 of Moyer (2003), and its original implementation only considers the bending effect of the Sun. Later in that section, the author indicates that ignoring the bending effect of the other massive bodies is a simplification. Since the computation of the bending effect is trivial, and considering it just for the Sun would increase the complexity of the function, this implementation takes it into account for all the massive bodies. The user can disable this effect using the `consider_bending` argument.

    In the description of the arguments, M is the number of massive bodies, N is the number of epochs, and the tuples at the end of the argument names indicate the expected shape of the arrays.

    Source: Moyer, Theodore D. (2003). Formulation for Observed and Computed Values of Deep Space Network Data Types for Navigation.

    :param bodies_gm: Array with GM of all the massive bodies to be considered (M,).
    :param x_obs_bcrf_rx: BCRF position of the observer at RX epochs (N, 3).
    :param x_src_bcrf_tx: BCRF position of the source at TX epochs (N, 3).
    :param x_bodies_bcrf_rx: BCRF position of the massive bodies at RX epochs (M, N, 3).
    :param x_bodies_bcrf_tx: BCRF position of the massive bodies at TX epochs (M, N, 3).
    :param consider_bending: Whether to consider the bending of the light path.
    :return: Post-Newtonian effect on the light travel time between source and observer (N,).
    """

    # Relative position of source and bodies at TX
    x_src_bodies_bcrf = (
        x_src_bcrf_tx[None, :, :] - x_bodies_bcrf_tx
    )  # (M, N, 3)
    r_src_bodies = np.linalg.norm(x_src_bodies_bcrf, axis=-1)  # (M, N)
    r0b = x_src_bodies_bcrf
    r0b_mag = r_src_bodies

    # Relative position of observer and bodies at RX
    x_obs_bodies_bcrf = (
        x_obs_bcrf_rx[None, :, :] - x_bodies_bcrf_rx
    )  # (M, N, 3)
    r_obs_bodies = np.linalg.norm(x_obs_bodies_bcrf, axis=-1)  # (M, N)
    r1b = x_obs_bodies_bcrf
    r1b_mag = r_obs_bodies

    # Difference of relative positions with respect to the bodies
    r01b_mag = np.linalg.norm(r1b - r0b, axis=-1)  # (M, N)

    # Reused term: 2 * GM / c^2
    gmc = 2.0 * bodies_gm[:, None] / (CLIGHT * CLIGHT)  # (M, 1)

    # Post-Newtonian effect
    dt_pn: np.ndarray = np.sum(
        (gmc / CLIGHT)
        * np.log(
            (r0b_mag + r1b_mag + r01b_mag + gmc * consider_bending)
            / (r0b_mag + r1b_mag - r01b_mag + gmc * consider_bending)
        ),
        axis=0,
    )  # (N,)

    return dt_pn
