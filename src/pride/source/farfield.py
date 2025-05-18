from typing import TYPE_CHECKING
from astropy import coordinates, time
import numpy as np
import spiceypy as spice
from ..constants import J2000
from .core import Source
from .. import io

if TYPE_CHECKING:
    from ..experiment import Experiment, Observation


class FarFieldSource(Source):

    def __init__(
        self,
        name: str,
        observed_right_ascension: float,
        observed_declination: float,
    ) -> None:

        # Basic initialization
        super().__init__(name)
        self.is_farfield = True

        # Set observed coordinates as attributes
        self.observed_ra = observed_right_ascension
        self.observed_dec = observed_declination

        # Calculate pointing vector from observed coordinates
        self.observed_ks = np.array(
            [
                np.cos(observed_right_ascension) * np.cos(observed_declination),
                np.sin(observed_right_ascension) * np.cos(observed_declination),
                np.sin(observed_declination),
            ]
        )

        return None

    @staticmethod
    def from_experiment(exp: "Experiment", name: str) -> "Source":

        raise NotImplementedError("This method is deprecated")

        # Initialize source
        source = FarFieldSource(name)

        # Read coordinates from VEX file
        source_info = exp._Experiment__vex._Vex__content["SOURCE"][name]
        _coords = coordinates.SkyCoord(
            source_info["ra"], source_info["dec"], frame="icrs"
        )
        _ra = float(_coords.ra.to("rad").value)  # type: ignore
        _dec = float(_coords.dec.to("rad").value)  # type: ignore
        source.observed_ra = _ra
        source.observed_dec = _dec

        # Calculate pointing vector
        source.observed_ks = np.array(
            [
                np.cos(_ra) * np.cos(_dec),
                np.sin(_ra) * np.cos(_dec),
                np.sin(_dec),
            ]
        )

        return source

    def spherical_coordinates(
        self, obs: "Observation"
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, time.Time | None
    ]:
        """Aberrated spherical coordinates of source at observation epochs

        Calculates azimuth and elevation of source corrected for diurnal (motion of station due to Earth's rotation) and annual (motion of station due to Earth's orbit around the Sun) aberration. Under the assumption that the source remains static during the period of observation, the right ascension and declination are taken directly from the VEX file.

        Source: Spherical Astrometry (сферическая астрометрия), Zharov (2006) - Eq 5.105 [Available online as of 01/2025]
        """

        clight = spice.clight() * 1e3

        # Convert RX to ephemeris time
        et_rx: np.ndarray = (
            (obs.tstamps.tdb - J2000.tdb).to("s").value  # type: ignore
        )

        # Calculate BCRF velocity of station at RX
        searth_bcrf_rx = (
            np.array(spice.spkezr("EARTH", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        vearth_bcrf_rx = searth_bcrf_rx[:, 3:]
        vsta_bcrf_rx = vearth_bcrf_rx + obs.station.velocity(
            obs.tstamps, frame="icrf"
        )
        v_mag = np.linalg.norm(vsta_bcrf_rx, axis=-1)[:, None]
        v_unit = vsta_bcrf_rx / v_mag

        # Unit vector along non-aberrated pointing direction
        s0 = self.observed_ks[None, :]

        # Aberrated pointing direction [Equation 5.105 from Zharov (2006)]
        v_c = v_mag / clight
        gamma = 1.0 / np.sqrt(1.0 - v_c * v_c)
        s0_dot_n = np.sum(s0 * v_unit, axis=-1)[:, None]
        s_aber = (
            (s0 / gamma)
            + (v_c * v_unit)
            + ((gamma - 1.0) * s0_dot_n * v_unit / gamma)
        ) / (1.0 + v_c * s0_dot_n)
        s_aber_icrf = s_aber / np.linalg.norm(s_aber, axis=-1)[:, None]

        # Transform pointing direction from GCRF to SEU
        s_aber_itrf = obs.icrf2itrf @ s_aber_icrf[:, :, None]
        s_aber_seu = (obs.seu2itrf.swapaxes(-1, -2) @ s_aber_itrf).squeeze().T

        # Calculate azimuth and elevation
        el = np.arcsin(s_aber_seu[2])
        az = np.arctan2(s_aber_seu[1], -s_aber_seu[0])
        az += (az < 0.0) * 2.0 * np.pi

        # Calculate station-centric right ascension and declination
        ra = np.ones_like(az) * self.observed_ra
        dec = np.ones_like(el) * self.observed_dec

        return az, el, ra, dec, None
