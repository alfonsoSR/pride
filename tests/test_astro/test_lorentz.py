from pride import astro
import pytest
import numpy as np


@pytest.mark.parametrize(
    [
        "target_position_gcrf",
        "earth_position_bcrf",
        "earth_velocity_bcrf",
        "potential_at_geocenter",
        "tolerances",
    ],
    [
        (
            np.array(
                [
                    -4.4599432574530998e06,
                    3.0900624529084368e06,
                    -3.3417962030995172e06,
                ]
            )[None, :],
            np.array(
                [
                    -1.7747671810167030e10,
                    1.3366906381386803e11,
                    5.7932537982011810e10,
                ]
            )[None, :],
            np.array(
                [
                    -3.0051588247328218e04,
                    -3.4151603064968435e03,
                    -1.4819683212624650e03,
                ]
            )[None, :],
            np.array([9.0233603016751158e08]),
            (1e-4, 0.0),
        ),
        (
            np.array(
                [
                    -4.4599432574530998e06,
                    3.0900624529084368e06,
                    -3.3417962030995172e06,
                ]
            )[None, :],
            np.array([0.0, 0.0, 0.0])[None, :],
            np.array(
                [
                    -3.0051588247328218e04,
                    -3.4151603064968435e03,
                    -1.4819683212624650e03,
                ]
            )[None, :],
            np.array([9.0233603016751158e08]),
            (0.0, 1e-15),
        ),
    ],
    ids=[
        "With Earth term",
        "Without Earth term",
    ],
)
def test_transformations_between_gcrf_and_bcrf(
    target_position_gcrf: np.ndarray,
    earth_position_bcrf: np.ndarray,
    earth_velocity_bcrf: np.ndarray,
    potential_at_geocenter: np.ndarray,
    tolerances: tuple[float, float],
) -> None:
    """Test the transformation between GCRF and BCRF coordinates

    Uses back conversion to check the Lorentz transformations between GCRF and BCRF coordinates. Could be improved with cross-validation against an external library (CALC?)

    The first case includes the Earth position in BCRF, which reduces the accuracy of the transformation due to the large differences in scale between geocentric and barycentric positions. For this test, sub-millimeter accuracy is required.

    The second case excludes the Earth position in BCRF, representing transformations between GCRS and geocentric BCRS. For this test, a relative error of 1e-15 is meant to ensure that the transformations preserve numerical precision.
    """

    calculated_bcrf = astro.transform_position_from_gcrf_to_bcrf(
        target_position_gcrf,
        earth_position_bcrf,
        earth_velocity_bcrf,
        potential_at_geocenter,
    )
    calculated_gcrf = astro.transform_position_from_bcrf_to_gcrf(
        calculated_bcrf,
        earth_position_bcrf,
        earth_velocity_bcrf,
        potential_at_geocenter,
    )

    atol, rtol = tolerances

    assert np.allclose(
        calculated_gcrf, target_position_gcrf, atol=atol, rtol=rtol
    )

    return None
