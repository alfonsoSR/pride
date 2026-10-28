from typing import TYPE_CHECKING
import numpy as np
from .. import astro, utils
from ..logger import log
from ..constants import CLIGHT
from ..source import FarFieldSource, NearFieldSource

if TYPE_CHECKING:
    from ..experiment.observation import Observation
    from ..delays.models.geometric import Geometric


UVW_METHODS: tuple[str, ...] = ("numerical", "analytic")
"""Available methods to calculate UVW projections"""

ANGULAR_STEP: float = 1e-4
"""Default angular displacement of the source for numerical derivatives [rad]"""


def projection_axes(
    ra: np.ndarray, dec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors of the UVW frame for a source direction

    :param ra: Right ascension of the source [rad] (N,)
    :param dec: Declination of the source [rad] (N,)
    :return e_ra: Unit vector along increasing right ascension (N, 3)
    :return e_dec: Unit vector along increasing declination (N, 3)
    :return s: Unit vector pointing to the source (N, 3)
    """

    zero = np.zeros_like(ra)
    e_ra = np.array([-np.sin(ra), np.cos(ra), zero]).T
    e_dec = np.array(
        [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]
    ).T
    s = np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    ).T

    return e_ra, e_dec, s


def calculate_uvw_analytic(obs: "Observation") -> np.ndarray:
    """UVW projections from the geometry of the baseline

    Projects the baseline vector (GCRF position of the station, since the phase center is the geocenter) on the UVW frame defined by the right ascension and declination of the source at the observation epochs. For a near-field source, these are the geocentric coordinates of the spacecraft (aberrated, as calculated by `NearFieldSource.spherical_coordinates`).

    The signs follow the convention of the geometric delay (tau ~ -b·s/c), so that w ~ c·tau, u ~ c·(dtau/dra)/cos(dec) and v ~ c·(dtau/ddec), as in the numerical method.

    :param obs: Observation with source coordinates
    :return: UVW projections [m] (N, 3)
    """

    baseline = obs.station.location(obs.tstamps, frame="icrf")
    e_ra, e_dec, s = projection_axes(obs.source_ra, obs.source_dec)

    return -np.array(
        [
            np.sum(baseline * e_ra, axis=-1),
            np.sum(baseline * e_dec, axis=-1),
            np.sum(baseline * s, axis=-1),
        ]
    ).T


def calculate_uvw_numerical(
    obs: "Observation",
    geometric: "Geometric",
    angular_step: float = ANGULAR_STEP,
) -> np.ndarray:
    """UVW projections from numerical derivatives of the geometric delay

    Follows the approach of pypride (Duev): u and v are obtained from the change in the geometric delay when the source is displaced in right ascension and declination, and w from the geometric delay itself:
    - u = c · (dtau/dra) / cos(dec)
    - v = c · (dtau/ddec)
    - w = c · tau

    The derivatives are calculated with central differences. For a far-field source, the pointing vector is rotated. For a near-field source, the geocentric position of the spacecraft at the TX epochs is rotated, keeping its distance to the geocenter, which displaces the spacecraft by angular_step times its geocentric distance.

    Differences with pypride (validated against its output for experiment GR035, station Badary):
    - Central differences instead of one-sided differences. The one-sided difference has an error of about |b|·h/2 (~300 m for h = 1e-4 rad), the central one of about |b|·h^2/6 (~1 cm).
    - For near-field sources, the axes are defined by the apparent direction of the spacecraft (position at TX minus Earth at RX, as used for the RA and Dec of the source), instead of the geometric direction (spacecraft minus Earth at the same epoch). The two differ by ~1e-4 rad, i.e. up to ~600 m in u and v.
    - w = c·tau in metres for all sources (pypride: dtau/drho in s/m for near-field sources).

    :param obs: Observation with source coordinates
    :param geometric: Geometric delay model
    :param angular_step: Angular displacement of the source [rad]
    :return: UVW projections [m] (N, 3)
    """

    source = obs.source
    ra: np.ndarray = obs.source_ra
    dec: np.ndarray = obs.source_dec

    if isinstance(source, FarFieldSource):

        def delay(dra: float, ddec: float) -> np.ndarray:
            _ra = source.observed_ra + dra
            _dec = source.observed_dec + ddec
            ks = np.array(
                [
                    np.cos(_ra) * np.cos(_dec),
                    np.sin(_ra) * np.cos(_dec),
                    np.sin(_dec),
                ]
            )
            return geometric.calculate_farfield(obs, ks=ks)

    elif isinstance(source, NearFieldSource):

        # Geocentric position of the spacecraft at TX, with the Earth at RX
        # [Same vector used for the RA and Dec of the source]
        et_tx = utils.get_ephemeris_time_from_epoch(obs.tx_epochs)
        et_rx = utils.get_ephemeris_time_from_epoch(obs.tstamps)
        xsrc_bcrf_tx = astro.get_icrf_position_vector(source.spice_id, et_tx)
        xearth_bcrf_rx = astro.get_icrf_position_vector("earth", et_rx)
        rho = np.linalg.norm(xsrc_bcrf_tx - xearth_bcrf_rx, axis=-1)

        def delay(dra: float, ddec: float) -> np.ndarray:
            _, _, s = projection_axes(ra + dra, dec + ddec)
            return geometric.calculate_nearfield_sekido_fukushima(
                obs, xsrc_bcrf_tx=xearth_bcrf_rx + rho[:, None] * s
            )

    else:
        log.error("Failed to calculate UVW projections: Invalid source type")
        exit(1)

    # Central differences
    h = angular_step
    dtau_dra = (delay(h, 0.0) - delay(-h, 0.0)) / (2.0 * h)
    dtau_ddec = (delay(0.0, h) - delay(0.0, -h)) / (2.0 * h)
    tau = delay(0.0, 0.0)

    return CLIGHT * np.array([dtau_dra / np.cos(dec), dtau_ddec, tau]).T
