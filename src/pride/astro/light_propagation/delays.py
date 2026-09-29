import numpy as np
from ..gravitation import calculate_newtonian_potential_from_bcrf_positions
from ...constants import CLIGHT, L_C


def calculate_sekido_fukushima_near_field_delay(
    bodies_gm: np.ndarray,
    x_obs_bcrf_rx: np.ndarray,
    x_obs_gcrf_rx: np.ndarray,
    s_earth_bcrf_rx: np.ndarray,
    x_src_bcrf_tx: np.ndarray,
    x_bodies_bcrf_rx: np.ndarray,
    relativistic_correction: np.ndarray,
) -> np.ndarray:
    """Near-field geometric delay - Sekido & Fukushima

    This function calculates the geometric delay between the times of arrival of a signal travelling from a near-field source to an observer and the geocenter using the mathematical model derived in Sekido & Fukushima 2006.

    The delay is returned in TT scale, and represents the difference between the time of arrival to the geocenter and the one to the station, meaning that it is positive when the station is closer to the source.

    In the description of the arguments, M is the number of **external** massive bodies, N is the number of epochs, and the tuples at the end of the argument names indicate the expected shape of the arrays.

    Source: Sekido, M., & Fukushima, T. (2006). A VLBI Delay Model for Radio Sources at a Finite Distance. Journal of Geodesy, 80(3), 137–149. https://doi.org/10.1007/s00190-006-0035-y

    :param bodies_gm: Array with GM of all the external massive bodies to be considered when calculating the gravitational potential at the geocenter. It should never contain the GM of the Earth. (M,)
    :param x_obs_bcrf_rx: BCRF position of the observer (station) at RX epochs. (N, 3)
    :param x_obs_gcrf_rx: GCRF position of the observer (station) at RX epochs. (N, 3)
    :param s_earth_bcrf_rx: BCRF state (i.e. position and velocity) of the geocenter at RX epochs. (N, 6).
    :param x_src_bcrf_tx: BCRF position of the source at TX epochs. (N, 3)
    :param x_bodies_bcrf_rx: BCRF position of the external massive bodies at RX epochs. (M, N, 3)
    :param relativistic_correction: Pre-computed relativistic corrections to be applied when calculating the near=field delay. (N,)
    :return: Estimated geometric delay between the geocenter and the observer (station) in TT scale. (N,)
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
        -(numerator_term_1 + numerator_term_2 + relativistic_correction)
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
    """Near-field geometric delay - Duev

    This function calculates the geometric delay between the times of arrival of a signal travelling from a near-field source to an observer and the geocenter using the mathematical model derived in Duev 2012.

    The delay is returned in TT scale and represents the difference between the time of arrival to the geocenter and the one to the station, meaning that it is positive when the station is closer to the source.

    In the description of the arguments, M is the number of external massive bodies, N is the number of epochs, and the tuples at the end of the argument names indicate the expected shape of the arrays.

    .. warning::
        The formulation presented here matches the implementation in the original version of this library, which slightly differs from the one presented in Duev 2012 in that the latter does not include the relativistic correction when transforming the delay to TT

    Source: Duev, D. A., Molera Calvés, G., Pogrebenko, S. V., Gurvits, L. I., Cimó, G., & Bocanegra Bahamon, T. (2012). Spacecraft VLBI and Doppler tracking: Algorithms and implementation. Astronomy & Astrophysics, 541, A43. https://doi.org/10.1051/0004-6361/201218885

    :param reference_delay: Pre-computed difference between times of arrival of the signal to the geocenter and the observer (station), in TDB scale. (N,)
    :param bodies_gm: Array with GM of all the external massive bodies to be considered when calculating the gravitational potential at the geocenter. It should not contain the Earth. (M,)
    :param x_obs_gcrf_rx: GCRF position vector of the observer (station) at RX epochs. (N, 3)
    :param s_earth_bcrf_rx: BCRF state vector (i.e. position and velocity) of the geocenter at RX epochs. (N, 6)
    :param x_bodies_bcrf_rx: BCRF position vector of the external massive bodies at RX epochs. (M, N, 3)
    :param relativistic_correction: Pre-computed relativistic corrections to be applied when calculating the near=field delay. (N,)
    :return: Estimated geometric delay between the geocenter and the observer (station) in TT scale. (N,)
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
