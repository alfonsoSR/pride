from pride.astro import lorentz_transformations as lorentz
import pytest
import numpy as np
from pride.constants import L_C, CLIGHT
import spiceypy as spice
from pride import io

TOL = 3e-16


def test_transform_position_from_gcrf_to_bcrf() -> None:

    target_position = np.array([[1, 0, 0]])
    earth_position = np.array([[1, 0, 0]])
    earth_velocity = np.array([[1, 0, 0]])
    potential = np.array([1])

    # Set target and velocity terms to zero -> Should return earth_bcrf
    expected_earth = lorentz.transform_position_from_gcrf_to_bcrf(
        target_position * 0, earth_position, earth_velocity * 0, potential * 0
    )
    assert np.all(np.isclose(expected_earth, earth_position, rtol=TOL))

    # Set earth and velocity to zero -> Should return target * (1 - Lc)
    expected_target = lorentz.transform_position_from_gcrf_to_bcrf(
        target_position, earth_position * 0, earth_velocity * 0, potential * 0
    )
    assert np.all(
        np.isclose(target_position * (1 - L_C), expected_target, rtol=TOL)
    )

    # Missing some test for the general case!

    return None
