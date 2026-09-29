import pytest
import numpy as np
from pride.astro.relativity import gravitational_effects as ge


@pytest.mark.parametrize(
    [
        "sun_gm",
        "x_obs_bcrf_rx",
        "x_src_bcrf_tx",
        "x_sun_bcrf_rx",
        "x_sun_bcrf_tx",
        "expected_dtpn",
        "tolerances",
    ],
    [
        (
            np.array([1.3271244004193930e20]),
            np.array(
                [
                    1.3313608831716026e11,
                    5.8760257524270782e10,
                    2.5505984680594021e10,
                ]
            )[None, :],
            np.array(
                [
                    1.2742132271587457e11,
                    3.9376799895709412e10,
                    1.7182577931667103e10,
                ]
            )[None, :],
            np.array(
                [
                    -1.2420246082963057e09,
                    -3.3706490003028923e08,
                    -1.1138769270566390e08,
                ]
            )[None, None, :],
            np.array(
                [
                    -1.2420251398830345e09,
                    -3.3706400803637427e08,
                    -1.1138730172159752e08,
                ]
            )[None, None, :],
            np.array([1.5150177202252381e-6]),
            (0.0, 1e-15),
        )
    ],
)
def test_post_newtonian_near_field_effect(
    sun_gm: np.ndarray,
    x_obs_bcrf_rx: np.ndarray,
    x_src_bcrf_tx: np.ndarray,
    x_sun_bcrf_rx: np.ndarray,
    x_sun_bcrf_tx: np.ndarray,
    expected_dtpn: np.ndarray,
    tolerances: tuple[float, float],
) -> None:
    """Test PN near-field effect calculation

    The bending term is disabled because it is not included in the TGRAV subroutine of CALC11

    Input and expected output generated using DIFXCALC11 (15/07/2025)
    """

    dt_pn = ge.post_newtonian_near_field_effect(
        sun_gm,
        x_obs_bcrf_rx,
        x_src_bcrf_tx,
        x_sun_bcrf_rx,
        x_sun_bcrf_tx,
        consider_bending=False,
    )
    atol, rtol = tolerances
    assert np.allclose(dt_pn, expected_dtpn, atol=atol, rtol=rtol)

    return None
