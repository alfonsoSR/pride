import pytest
import numpy as np
from pride import astro


@pytest.mark.parametrize(
    ["bodies_gm", "distances", "expected"],
    [
        (
            np.array([1.0, 10.0, 100.0]),
            np.array([[1.0, 2.0, 50.0], [0.01, 5.0, 2.0]]).T,
            np.array([8.0, 152.0]),
        )
    ],
)
def test_potential_from_distances(bodies_gm, distances, expected) -> None:

    result = astro.calculate_newtonian_potential(bodies_gm, distances)
    assert np.allclose(result - expected, 0, atol=0, rtol=3e-16)

    return None


@pytest.mark.parametrize(
    ["bodies_gm", "x_target_bcrf", "x_bodies_bcrf", "expected"],
    [
        (
            np.array([10.0]),
            np.array([[12.0, -2.3, 50.0], [1.0, 0.0, 4.0]]),
            np.array([[11.0, -4.3, 0.0], [0.99, -5.0, 2.0]])[None, :, :],
            np.array([1.9980029950087341e-01, 1.8569501801350372e00]),
        )
    ],
)
def test_potential_from_bcrf_positions(
    bodies_gm, x_target_bcrf, x_bodies_bcrf, expected
) -> None:

    result = astro.calculate_newtonian_potential_from_bcrf_positions(
        bodies_gm, x_target_bcrf, x_bodies_bcrf
    )
    assert np.allclose(result - expected, 0, atol=0, rtol=3e-16)

    return None
