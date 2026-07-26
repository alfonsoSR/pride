import numpy as np
from ...constants import CLIGHT, L_C
from ..gravitation import calculate_newtonian_potential_from_bcrf_positions
from astropy import time


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

    # Relative position between source and observer
    # r01 = x_obs_bcrf_rx - x_src_bcrf_tx
    # r01_mag = np.linalg.norm(r01, axis=-1)

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


def post_newtonian_near_field_delay(
    bodies_gm: np.ndarray,
    x_obs1_bcrf_rx1: np.ndarray,
    x_obs2_bcrf_rx2: np.ndarray,
    x_src_bcrf_tx: np.ndarray,
    x_bodies_bcrf_rx1: np.ndarray,
    x_bodies_bcrf_rx2: np.ndarray,
    x_bodies_bcrf_tx: np.ndarray,
    consider_bending: bool = True,
) -> np.ndarray:
    """Difference in post-Newtonian gravitational effect due to moving solar system bodies for two observers of a common source

    This function calculates the difference between the contributions of post-newtonian relativistic effects to the light-time of a signal travelling from a near-field source to a couple of observers.

    Mathematically, this is equivalent to evaluating `astro.post_newtonian_near_field_effect` for both observers and then taking the difference, but floating point errors lead to non-negligible differences between these two approaches. The user is referred to the API entry of `astro.post_newtonian_near_field_effect` for more information about the modeling of the relativistic effect.

    In the description of the arguments, M is the number of massive bodies, N is the number of epochs, and the tuples at the end of the argument names indicate the expected shape of the arrays.

    :param bodies_gm: Array with GM of all the massive bodies to be considered (M,).
    :param x_obs1_bcrf_rx1: BCRF position of the first observer at its RX epochs (N, 3).
    :param x_obs2_bcrf_rx2: BCRF position of the second observer at its RX epochs (N, 3).
    :param x_src_bcrf_tx: BCRF position of the source at TX epochs (N, 3).
    :param x_bodies_bcrf_rx1: BCRF position of the massive bodies at the RX epochs of the first observer (M, N, 3).
    :param x_bodies_bcrf_rx2: BCRF position of the massive bodies at RX epochs of the second observer(M, N, 3).
    :param x_bodies_bcrf_tx: BCRF position of the massive bodies at TX epochs (M, N, 3).
    :param consider_bending: Whether to consider the bending of the light path.
    :return: Difference in post-Newtonian effect on the light travel times between source and observers 2 and 1 (N,).
    """

    # Relative position of source and bodies at TX
    x_src_bodies_bcrf = (
        x_src_bcrf_tx[None, :, :] - x_bodies_bcrf_tx
    )  # (M, N, 3)
    r_src_bodies = np.linalg.norm(x_src_bodies_bcrf, axis=-1)  # (M, N)
    r0b = x_src_bodies_bcrf
    r0b_mag = r_src_bodies

    # Relative position of observers and bodies at RX epochs
    x_obs1_bodies_bcrf = (
        x_obs1_bcrf_rx1[None, :, :] - x_bodies_bcrf_rx1
    )  # (M, N, 3)
    r_obs1_bodies = np.linalg.norm(x_obs1_bodies_bcrf, axis=-1)  # (M, N)
    r1b = x_obs1_bodies_bcrf
    r1b_mag = r_obs1_bodies

    x_obs2_bodies_bcrf = (
        x_obs2_bcrf_rx2[None, :, :] - x_bodies_bcrf_rx2
    )  # (M, N, 3)
    r_obs2_bodies = np.linalg.norm(x_obs2_bodies_bcrf, axis=-1)  # (M, N)
    r2b = x_obs2_bodies_bcrf
    r2b_mag = r_obs2_bodies

    # Relative position between source and observers
    r01 = x_obs1_bcrf_rx1 - x_src_bcrf_tx
    r01_mag = np.linalg.norm(r01, axis=-1)
    r02 = x_obs2_bcrf_rx2 - x_src_bcrf_tx
    r02_mag = np.linalg.norm(r02, axis=-1)

    # Difference of relative positions with respect to the bodies
    r01b_mag = np.linalg.norm(r1b - r0b, axis=-1)  # (M, N)
    r02b_mag = np.linalg.norm(r2b - r0b, axis=-1)  # (M, N)

    # Reused term: 2 * GM / c^2
    gmc = 2.0 * bodies_gm[:, None] / (CLIGHT * CLIGHT)  # (M, 1)

    # Post-Newtonian effect
    dt_pn: np.ndarray = np.sum(
        (gmc / CLIGHT)
        * np.log(
            (
                (r2b_mag + r0b_mag + r02b_mag + gmc * consider_bending)
                * (r1b_mag + r0b_mag - r01b_mag + gmc * consider_bending)
            )
            / (
                (r2b_mag + r0b_mag - r02b_mag + gmc * consider_bending)
                * (r1b_mag + r0b_mag + r01b_mag + gmc * consider_bending)
            )
        ),
        axis=0,
    )  # (N,)

    return dt_pn


def calculate_sekido_fukushima_near_field_delay(
    bodies_gm: np.ndarray,
    x_obs_bcrf_rx: np.ndarray,
    x_obs_gcrf_rx: np.ndarray,
    s_earth_bcrf_rx: np.ndarray,
    x_src_bcrf_tx: np.ndarray,
    x_bodies_bcrf_rx: np.ndarray,
    post_newtonian_correction: np.ndarray,
) -> np.ndarray:
    """Near-field, geometric delay between station and geocenter

    Calculates the geometric delay between the times of arrival of a signal travelling from a near-field radio source to an observer and the geocenter in TT scale following the mathematical model in Sekido and Fukusima 2006.
    """

    # Separate position and velocity of Earth in BCRF at RX
    x_earth_bcrf_rx = s_earth_bcrf_rx[:, :3]
    v_earth_bcrf_rx = s_earth_bcrf_rx[:, 3:]

    # Calculate pseudo source-pointing vector: k_vec
    x_obs_src_bcrf_rx = x_src_bcrf_tx - x_obs_bcrf_rx
    r_obs_src_bcrf_rx = np.linalg.norm(x_obs_src_bcrf_rx, axis=-1)
    x_earth_src_bcrf_rx = x_src_bcrf_tx - x_earth_bcrf_rx
    r_earth_src_bcrf_rx = np.linalg.norm(x_earth_src_bcrf_rx, axis=-1)
    k_vec = (x_obs_src_bcrf_rx + x_earth_src_bcrf_rx) / (
        r_obs_src_bcrf_rx + r_earth_src_bcrf_rx
    )[:, None]

    # Calculate external gravitational potential at geocenter
    U_geocenter = calculate_newtonian_potential_from_bcrf_positions(
        massive_bodies_gm=bodies_gm,
        x_target_bcrf=x_earth_bcrf_rx,
        x_bodies_bcrf=x_bodies_bcrf_rx,
    )

    # Convenience variables
    b_gcrf = -x_obs_gcrf_rx  # Baseline vector in GCRF
    clight2 = CLIGHT * CLIGHT

    # Calculate first term in numerator
    v_earth_squared = np.sum(v_earth_bcrf_rx * v_earth_bcrf_rx, axis=-1)
    k_dot_b = np.sum(k_vec * b_gcrf, axis=-1)
    numerator_term_1 = -(
        1.0 - ((2.0 * U_geocenter) + (0.5 * v_earth_squared)) / clight2
    ) * (k_dot_b / CLIGHT)

    # Calculate second term in numerator
    ve_dot_b = np.sum(v_earth_bcrf_rx * b_gcrf, axis=-1)
    k_dot_ve = np.sum(k_vec * v_earth_bcrf_rx, axis=-1)
    ux_earth_src_bcrf_rx = x_earth_src_bcrf_rx / r_earth_src_bcrf_rx[:, None]
    ux_earth_dot_ve = np.sum(ux_earth_src_bcrf_rx * v_earth_bcrf_rx, axis=-1)
    numerator_term_2 = -(ve_dot_b / clight2) * (
        1.0 + ((ux_earth_dot_ve - 0.5 * k_dot_ve) / CLIGHT)
    )

    # Calculate denominator
    ve_c_cross_ux_earth = np.cross(
        v_earth_bcrf_rx / CLIGHT, ux_earth_src_bcrf_rx
    )
    cross_norm_squared = np.sum(
        ve_c_cross_ux_earth * ve_c_cross_ux_earth, axis=-1
    )
    h = 0.5 * cross_norm_squared * k_dot_b / r_earth_src_bcrf_rx
    denominator = (1.0 + (ux_earth_dot_ve / CLIGHT)) * (1.0 + h)

    # Return delay in TT
    return (
        -(numerator_term_1 + numerator_term_2 + post_newtonian_correction)
        / denominator
    )


def calculate_duev_near_field_delay(
    reference_delay: np.ndarray,
    bodies_gm: np.ndarray,
    x_obs_gcrf_rx: np.ndarray,
    s_earth_bcrf_rx: np.ndarray,
    x_bodies_bcrf_rx: np.ndarray,
    relativistic_correction: np.ndarray,
) -> np.ndarray:
    """Near-field, geometric delay between station and geocenter

    Calculates the geometric delay between the times of arrival of a signal travelling from a near-field radio source to an observer and the geocenter in TT scale using the mathematical model in Duev 2012.

    NOTE: With respect to the formulation in the paper, the denominator is equal to zero because the second ground station is the geocenter
    """

    # Separate position and velocity of Earth
    x_earth_bcrf_rx = s_earth_bcrf_rx[:, :3]
    v_earth_bcrf_rx = s_earth_bcrf_rx[:, 3:]

    # Calculate external gravitational potential at geocenter
    U_geocenter = calculate_newtonian_potential_from_bcrf_positions(
        bodies_gm, x_earth_bcrf_rx, x_bodies_bcrf_rx
    )

    # Calculate intermediate quantities
    clight2 = CLIGHT * CLIGHT
    v_earth_squared = np.sum(v_earth_bcrf_rx * v_earth_bcrf_rx, axis=-1)
    v_earth_dot_b = np.sum(-v_earth_bcrf_rx * x_obs_gcrf_rx, axis=-1)

    return (
        (reference_delay + relativistic_correction)
        * (1.0 - (0.5 * v_earth_squared + U_geocenter) / clight2)
        / (1.0 - L_C)
    ) - (v_earth_dot_b / clight2)
