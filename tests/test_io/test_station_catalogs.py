from pride import io
import pytest
import numpy as np
from astropy import time

TOL = 1e-15


@pytest.mark.parametrize(
    ["station", "expected_coordinates", "fails"],
    [
        (
            "YAMAGU32",
            np.array([-3502544.259, 3950966.397, 3566381.165]),
            False,
        ),  # YAMAGUCHI
        (
            "INVALID",
            np.array([0.0, 0.0, 0.0]),
            True,
        ),  # Fails: Invalid station name
    ],
    ids=[
        "Yamaguchi",
        "Invalid station name",
    ],
)
def test_get_station_coordinates(
    station: str, expected_coordinates: np.ndarray, fails: bool
) -> None:

    if fails:
        with pytest.raises(SystemExit):
            _ = io.load_station_coordinates_from_catalog(station)
        return None

    coords = io.load_station_coordinates_from_catalog(station)
    assert np.allclose(coords, expected_coordinates, atol=TOL)

    return None


@pytest.mark.parametrize(
    ["station", "expected_velocity", "fails"],
    [
        (
            "WSTRBORK",
            np.array([-13.53e-3, 17.04e-3, 8.73e-3]),
            False,
        ),  # YAMAGUCHI
        (
            "INVALID",
            np.array([0.0, 0.0, 0.0]),
            True,
        ),  # Fails: Invalid station name
    ],
    ids=[
        "WSTRBORK",
        "Invalid station name",
    ],
)
def test_get_station_velocity(
    station: str, expected_velocity: np.ndarray, fails: bool
) -> None:

    if fails:
        with pytest.raises(SystemExit):
            _ = io.load_station_velocity_from_catalog(station)
        return None

    v = io.load_station_velocity_from_catalog(station)
    assert np.allclose(v, expected_velocity, atol=TOL)

    return None


def test_get_reference_epoch() -> None:

    expected_epoch = time.Time("2000-01-01T00:00:00", scale="utc")
    reference_epoch = io.load_reference_epoch_for_station_catalog()
    difference = (reference_epoch - expected_epoch).to_value("s")
    assert isinstance(difference, float)
    assert abs(difference) == 0

    return None
