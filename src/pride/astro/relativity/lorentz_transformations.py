import numpy as np
from ...constants import L_C, CLIGHT, GAMMA_PPN

def transform_position_from_gcrf_to_bcrf(
    x_target_gcrf: np.ndarray,
    x_earth_bcrf: np.ndarray,
    v_earth_bcrf: np.ndarray,
    potential_at_geocenter: np.ndarray,
) -> np.ndarray:
    """Lorentz transformation of position vector from GCRF to BCRF

    Source: 10.1051/0004-6361/201218885 (Duev 2012)

    :param x_target_gcrf: GCRF position of the target as (N, 3) array
    :param x_earth_bcrf: BCRF positon of the Earth as (N, 3) array
    :param v_earth_bcrf: BCRF velocity of the Earth as (N, 3) array
    :param potential_at_geocenter: Newtonian potential of all the solar system bodies, excluding the Earth, evaluated at the geocenter. Passed as (N,) array.
    :return: BCRF position of the target as (N, 3) array
    """

    # Precomputed quantities
    clight2 = CLIGHT * CLIGHT

    # Components of the expression
    term1 = (1.0 - L_C - (GAMMA_PPN * potential_at_geocenter / clight2))[
        :, None
    ] * x_target_gcrf
    term2 = (
        0.5
        * np.sum(v_earth_bcrf * x_target_gcrf, axis=-1)[:, None]
        * v_earth_bcrf
        / clight2
    )
    term3 = x_earth_bcrf

    return term1 - term2 + term3


def transform_position_from_bcrf_to_gcrf(
    x_target_bcrf: np.ndarray,
    x_earth_bcrf: np.ndarray,
    v_earth_bcrf: np.ndarray,
    potential_at_geocenter: np.ndarray,
) -> np.ndarray:
    """Lorentz transformation of position vector from BCRF to GCRF

    Source: Moyer, Theodore D.. Formulation for Observed and Computed Values of Deep Space Network Data Types for Navigation. (2003).

    :param x_target_bcrf: BCRF position of the target as (N, 3) array
    :param x_earth_bcrf: BCRF position of the Earth as (N, 3) array
    :param v_earth_bcrf: BCRF velocity of the Earth as (N, 3) array
    :param potential_at_geocenter: Newtonian potential of all the solar system bodies, excluding the Earth, evaluated at the geocenter. Passed as (N,) array.
    :return: GCRF position of the target as (N, 3) array
    """

    # Precomputed quantities
    clight2 = CLIGHT * CLIGHT

    # Relative position of the target with respect to the Earth in BCRF metric
    x_target_earth_bcrf = x_target_bcrf - x_earth_bcrf

    # Components of the expression
    term1 = (1.0 + L_C + (GAMMA_PPN * potential_at_geocenter / clight2))[
        :, None
    ] * x_target_earth_bcrf
    term2 = (
        0.5
        * np.sum(v_earth_bcrf * x_target_earth_bcrf, axis=-1)[:, None]
        * v_earth_bcrf
        / clight2
    )

    return term1 + term2
