from typing import TYPE_CHECKING
from astropy import time
from ..logger import log
import numpy as np
from .. import io
from .core import Source
from .. import astro, utils
from ..constants import CLIGHT

if TYPE_CHECKING:
    from ..experiment import Experiment, Observation
    from ..station import Station


class NearFieldSource(Source):

    def __init__(self, name: str) -> None:

        # Basic initialization
        super().__init__(name)
        self.is_nearfield = True

        return None

    @staticmethod
    def from_experiment(exp: "Experiment", name: str = "") -> "Source":

        # Initialize source
        source = NearFieldSource(exp.target["short_name"])
        source.spice_id = exp.target["short_name"]

        # Load ramping data for three-way link
        _3way_source = io.get_path_to_ramping_data_file(
            source.name, "three-way"
        )
        # path_base: str = exp.setup.catalogues["frequency_ramping"]
        # _3way_source = io.internal_file(f"{path_base}3w.{source.spice_id}")
        _3way_data = io.load_ramping_data(
            _3way_source, "three-way", (exp.initial_epoch, exp.final_epoch)
        )
        if _3way_data is not None:
            source.three_way_ramping = _3way_data
            source.has_three_way_ramping = True
        else:
            log.warning(f"Three-way ramping data not found for {source.name}")

        # Load ramping data for one-way link
        _1way_source = io.get_path_to_ramping_data_file(source.name, "one-way")
        # _1way_source = io.internal_file(f"{path_base}1w.{source.spice_id}")
        _1way_data = io.load_ramping_data(
            _1way_source, "one-way", (exp.initial_epoch, exp.final_epoch)
        )
        if _1way_data is not None:
            source.one_way_ramping = _1way_data
            source.has_one_way_ramping = True
        else:
            log.warning(f"One-way ramping data not found for {source.name}")

        # Set default downlink frequency
        source.default_frequency = exp.target["downlink_frequency"] * 1e6  # Hz

        return source

    def tx_from_rx(self, rx: "time.Time", station: "Station") -> time.Time:
        """Calculate TX epoch from RX epoch at a station"""

        # Sanity
        if station.is_phase_center:
            log.error(
                "Calculation of TX from RX not valid for station at geocenter"
            )
            exit(1)

        # Calculate GCRF coordinates of station at RX
        xsta_gcrf_rx = station.location(rx, frame="icrf")

        # Calculate BCRS position of source at RX
        et_rx = utils.get_ephemeris_time_from_epoch(rx)

        # Get list of bodies to consider for relativistic correction
        _bodies = io.internal_parameter("lt_correction_bodies")
        external_bodies = [body for body in _bodies if body.lower() != "earth"]
        bodies = external_bodies + ["earth"]

        # Initialize arrays for GM and BCRF position of massive bodies
        x_external_bodies_bcrf_rx = np.zeros(
            (len(external_bodies), len(et_rx), 3)
        )
        external_bodies_gm = np.zeros(len(external_bodies))

        # Get GM and BCRF position of all external bodies at RX
        for idx, body in enumerate(external_bodies):

            external_bodies_gm[idx] = astro.get_body_gravitational_parameter(
                body
            )
            x_external_bodies_bcrf_rx[idx] = astro.get_icrf_position_vector(
                body, et_rx
            )

        # Add Earth to the arrays including all bodies and get its BCRF velocity
        __s_earth_bcrf_rx = astro.get_icrf_state_vector("earth", et_rx)
        xearth_bcrf_rx = __s_earth_bcrf_rx[:, :3]
        vearth_bcrf_rx = __s_earth_bcrf_rx[:, 3:]

        # Calculate Newtonian potential of external bodies at geocenter
        U_earth = astro.calculate_newtonian_potential_from_bcrf_positions(
            massive_bodies_gm=external_bodies_gm,
            x_target_bcrf=xearth_bcrf_rx,
            x_bodies_bcrf=x_external_bodies_bcrf_rx,
        )

        # Calculate BCRS position of station at RX
        xsta_bcrf_rx = astro.transform_position_from_gcrf_to_bcrf(
            xsta_gcrf_rx,
            xearth_bcrf_rx,
            vearth_bcrf_rx,
            U_earth,
        )

        # Retrieve settings for Newton-Raphson algorithm
        precision = float(io.internal_parameter("lt_precision"))
        n_max = int(io.internal_parameter("lt_max_iterations"))

        # Calculate light-time between source and station
        lt_0 = astro.light_time_from_rx_epoch(
            rx, xsta_bcrf_rx, self.spice_id, bodies, precision, n_max
        )

        return rx.tdb - lt_0

    def spherical_coordinates(
        self, obs: "Observation"
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, time.Time | None
    ]:

        # Calculate TX epochs for observation
        tx = self.tx_from_rx(obs.tstamps, obs.station)

        # Convert TX and RX epochs to ephemeris time
        et_tx = utils.get_ephemeris_time_from_epoch(tx)
        et_rx = utils.get_ephemeris_time_from_epoch(obs.tstamps)

        # Calculate aberrated position of source wrt station
        xsrc_gcrf_tx = astro.get_gcrf_position_vector(self.spice_id, et_tx)
        xsrc_sta_ab = xsrc_gcrf_tx - obs.station.location(obs.tstamps, "icrf")

        # Calculate aberrated position of source wrt Earth
        xsrc_bcrf_tx = astro.get_icrf_position_vector(self.spice_id, et_tx)
        xearth_bcrf_rx = astro.get_icrf_position_vector("earth", et_rx)
        xsrc_earth_ab = (xsrc_bcrf_tx - xearth_bcrf_rx).T

        # Calculate aberrated pointing vector in SEU [az, el]
        k_gcrf = xsrc_sta_ab / np.linalg.norm(xsrc_sta_ab, axis=-1)[:, None]
        k_itrf = obs.icrf2itrf @ k_gcrf[:, :, None]
        s, e, u = (obs.seu2itrf.swapaxes(-1, -2) @ k_itrf).squeeze().T

        # Calculate azimuth and elevation
        az = np.arctan2(e, -s)
        az += (az < 0.0) * 2.0 * np.pi
        el = np.arcsin(u)

        # Calculate right ascension and declination
        ra = np.arctan2(xsrc_earth_ab[1], xsrc_earth_ab[0])
        ra += (ra < 0.0) * 2.0 * np.pi
        dec = np.arctan2(
            xsrc_earth_ab[2],
            np.sqrt(xsrc_earth_ab[0] ** 2 + xsrc_earth_ab[1] ** 2),
        )

        return az, el, ra, dec, tx
