from pride import astro
import pytest
import numpy as np
from astropy import time
from pathlib import Path
import spiceypy as spice

DATA_DIRECTORY: Path = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize(
    [
        "tdb_rx",
        "xrec_bcrf_rx",
        "source_id",
        "perturbing_bodies",
        "precision",
        "max_iterations",
        "expected",
    ],
    [
        (
            time.Time(["2013-12-28 18:18:07.183839"], scale="tdb"),
            np.array(
                [
                    [
                        -1.7752131753292191e10,
                        1.3367215387624660e11,
                        5.7929196185892807e10,
                    ]
                ],
            ),
            "mars",
            [
                "sun",
                "earth",
            ],
            1e-15,
            6,
            697.1767116867476,
        ),
    ],
    ids=[
        "Default",
    ],
)
def test_light_time_solution(
    tdb_rx: "time.Time",
    xrec_bcrf_rx: np.ndarray,
    source_id: str,
    perturbing_bodies: list[str],
    precision: float,
    max_iterations: int,
    expected: time.TimeDelta,
) -> None:
    """Test the solution of the light-time equation

    Ensures consistency with previous version of the program to sub-ns level
    """

    kdir = DATA_DIRECTORY / "kernels"
    with spice.KernelPool(
        [str(kdir / "DE405.BSP"), str(kdir / "DE403-MASSES.TPC")]
    ):

        calculated = (
            astro.light_time_from_rx_epoch(
                tdb_rx,
                xrec_bcrf_rx,
                source_id,
                perturbing_bodies,
                precision,
                max_iterations,
            )
            .to("s")
            .value[0]
        )

    assert np.isclose(calculated, expected, rtol=1e-16, atol=1e-10)  # type: ignore

    return None


@pytest.mark.parametrize(
    [
        "tdb_tx",
        "xsrc_bcrf_tx",
        "receiver_id",
        "perturbing_bodies",
        "precision",
        "max_iterations",
        "expected",
    ],
    [
        (
            time.Time(["2013-12-28 18:06:29.996705"], scale="tdb"),
            np.array(
                [
                    [
                        -2.2345191254105060e11,
                        9.7331104953007004e10,
                        5.0668972643661644e10,
                    ]
                ],
            ),
            "earth",
            [
                "sun",
                "mercury_barycenter",
                "venus_barycenter",
                "moon",
                "mars_barycenter",
                "jupiter_barycenter",
                "saturn_barycenter",
                "uranus_barycenter",
                "neptune_barycenter",
            ],
            1e-15,
            6,
            697.200368142704,
        ),
    ],
    ids=[
        "Default",
    ],
)
def test_light_time_from_tx(
    tdb_tx: "time.Time",
    xsrc_bcrf_tx: np.ndarray,
    receiver_id: str,
    perturbing_bodies: list[str],
    precision: float,
    max_iterations: int,
    expected: time.TimeDelta,
) -> None:
    """Test the solution of the light-time equation

    Ensures consistency with previous version of the program to sub-ns level
    """

    kdir = DATA_DIRECTORY / "kernels"
    with spice.KernelPool(
        [str(kdir / "DE405.BSP"), str(kdir / "DE403-MASSES.TPC")]
    ):

        calculated = (
            astro.light_time_from_tx_epoch(
                tdb_tx,
                xsrc_bcrf_tx,
                receiver_id,
                perturbing_bodies,
                precision,
                max_iterations,
            )
            .to("s")
            .value[0]
        )

    assert np.isclose(calculated, expected, rtol=1e-16, atol=1e-10)  # type: ignore

    return None
