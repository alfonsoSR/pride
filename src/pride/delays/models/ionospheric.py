from ..core import Delay
from typing import TYPE_CHECKING, Any
from ...logger import log
from astropy import time
from ftplib import FTP_TLS
import unlzw3
import gzip
import numpy as np
from scipy import interpolate
from ... import astro, io, utils

if TYPE_CHECKING:
    from pathlib import Path
    from ...experiment.observation import Observation
    from ...station.core import Station


DEFAULT_THREE_WAY_BANDS: tuple[str, str] = ("X", "X")
"""Uplink and downlink bands of three-way links if not set in the Doppler section"""


class Ionospheric(Delay):
    """Ionospheric delay

    # Available models
    ## [MISSING NAME OF MODEL]
    - BRIEF DESCRIPTION OF THE MODEL AND SOURCE
    - DESCRIPTION OF REQUIRED RESOURCES AND KEYS
    """

    def three_way_turnaround_ratio(self) -> float:
        """Turnaround ratio of the three-way link of the near-field target

        The uplink and downlink bands are read from the Doppler section of the configuration file (uplink_band, downlink_band), so that the delays and the Doppler predictions use the same link. If they are not set, an X/X link is assumed.

        :return: Ratio of downlink and uplink frequencies
        """

        if "turnaround_ratio" in self.loaded_resources:
            return self.loaded_resources["turnaround_ratio"]

        doppler_setup = self.exp.setup.doppler
        if "uplink_band" in doppler_setup and "downlink_band" in doppler_setup:
            bands = (doppler_setup["uplink_band"], doppler_setup["downlink_band"])
        else:
            bands = DEFAULT_THREE_WAY_BANDS
            log.warning(
                "Uplink and downlink bands not set in the Doppler section: "
                f"Assuming {bands[0]}/{bands[1]} three-way link for the "
                "ionospheric delay"
            )

        ratio = astro.get_turnaround_ratio(*bands)
        if ratio is None:
            log.error(
                "Failed to calculate ionospheric delay: Turnaround ratio not "
                f"available for {bands[0]}/{bands[1]} uplink/downlink bands"
            )
            exit(1)

        self.loaded_resources["turnaround_ratio"] = ratio
        return ratio

    def ensure_resources(self) -> None:

        log.warning(
            "The implementation of the ionospheric delay should be revised"
        )

        # Define range of dates to look for ionospheric data
        date = utils.get_date_from_epoch(self.exp.initial_epoch)
        step = time.TimeDelta(1.0, format="jd")

        coverage: dict[tuple[time.Time, time.Time], Path] = {}
        while True:

            # Download ionex file for the given date
            ionex_file = io.download_ionex_file_for_date(
                date, self.config["data"]
            )

            # Add coverage to dictionary
            coverage[(date, date + step)] = ionex_file

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage to resources
        self.resources["coverage"] = coverage

        return None

    def load_resources(self) -> dict[str, Any]:

        # Generate TEC maps
        tec_grid_interpolators: dict[
            time.Time, interpolate.RegularGridInterpolator
        ] = {}

        # Reference values for Earth radius and ionospheric height
        reference_earth_radius: float = NotImplemented
        reference_ionospheric_height: float = NotImplemented

        # Load TEC maps from all the IONEX files
        for source in self.resources["coverage"].values():

            # Read IONEX file
            ionex_content = io.IonexInterface(source)
            tec_interpolators = ionex_content.generate_tec_map_interpolators()

            # Update reference Earth radius or check consistency
            if reference_earth_radius is NotImplemented:
                reference_earth_radius = ionex_content.ref_rearth
            elif reference_earth_radius != ionex_content.ref_rearth:
                log.error(
                    f"Failed to load TEC maps: "
                    "Inconsistent reference Earth radius across files"
                )
                exit(1)

            # Update reference ionospheric height or check consistency
            if reference_ionospheric_height is NotImplemented:
                reference_ionospheric_height = ionex_content.ref_height
            elif reference_ionospheric_height != ionex_content.ref_height:
                log.error(
                    f"Failed to load TEC maps: "
                    "Inconsistent reference ionospheric height across files"
                )
                exit(1)

            # Update list of TEC map interpolators
            for epoch, interpolator in tec_interpolators.items():
                tec_grid_interpolators[epoch] = interpolator

        # Initialize resources dictionary with reference values
        resources: dict[str, Any] = {
            "ref_height": reference_ionospheric_height,
            "ref_rearth": reference_earth_radius,
        }

        # Group TEC epochs into single time.Time object
        tec_epochs = time.Time(list(tec_grid_interpolators.keys()))

        # Keep TEC maps to load stations that are not part of the experiment
        self.tec_epochs = tec_epochs
        self.tec_grid_interpolators = tec_grid_interpolators

        # Generate a 1D TEC interpolator for each station
        for baseline in self.exp.baselines:
            resources[baseline.station.name] = self.load_station_resources(
                baseline.station
            )

            # # Generate a 1D interpolator with the TEC at station coordinates
            # # as function of the observation epoch
            # interp_type: str = (
            #     "linear" if len(tec_at_station_coordinates) <= 3 else "cubic"
            # )
            # resources[baseline.station.name] = interpolate.interp1d(
            #     tec_epochs.mjd, tec_at_station_coordinates, kind=interp_type
            # )

        return resources

    def load_station_resources(self, station: "Station") -> Any:
        """Interpolator of the vertical TEC at a station as function of MJD

        The station coordinates are corrected for tectonic motion only (the TEC maps
        do not require a better accuracy). This also works for stations whose
        position interpolators only cover a short time interval.

        :param station: Station object
        :return: Interpolator of the vertical TEC [0.1 TECU] at the station
        """

        tec_epochs = self.tec_epochs

        # Station latitude and longitude for each coverage epoch
        # TODO: REPLACE WITH GEOCENTRIC COORDINATES!!
        coords = station.tectonic_corrected_location(tec_epochs).to_geodetic(
            "GRS80"
        )
        station_latitudes: np.ndarray = coords.lat.deg  # type: ignore
        station_longitudes: np.ndarray = coords.lon.deg  # type: ignore

        # Get TEC at station coordinates for each epoch in coverage
        station_local_tec: list[float] = [
            float(tec_interpolator([lon, lat])[0])
            for tec_interpolator, lon, lat in zip(
                self.tec_grid_interpolators.values(),
                station_longitudes,
                station_latitudes,
            )
        ]

        # Interpolate local TEC as function of time
        return interpolate.interp1d(
            tec_epochs.mjd, station_local_tec, kind="cubic"
        )

    def calculate(
        self, obs: "Observation", frequency: np.ndarray | None = None
    ) -> Any:
        """Ionospheric group delay [s]

        :param obs: Observation
        :param frequency: Frequency of the signal at each epoch [Hz]. If not given, it is determined from the observation band (far-field sources) or from the frequency ramping tables (near-field sources), as before. Used by the Doppler predictions to evaluate the delay at the uplink and downlink frequencies of each leg.
        :return: Ionospheric group delay at the observation epochs [s]
        """

        # Load resources for stations that are not part of the experiment
        # (e.g. uplink stations of a three-way Doppler link)
        if obs.station.name not in self.loaded_resources:
            self.loaded_resources[obs.station.name] = (
                self.load_station_resources(obs.station)
            )

        # Get vertical TEC from station at observation epochs
        vtec = self.loaded_resources[obs.station.name](obs.tstamps.mjd)

        # Read reference height and Earth radius from model
        h_ref = self.loaded_resources["ref_height"]
        r_earth = self.loaded_resources["ref_rearth"]

        # NOTE: Using model from Petrov (based on original code)
        zenith = np.arcsin(np.cos(obs.source_el) / (1.0 + (h_ref / r_earth)))
        tec = 0.1 * vtec / np.cos(zenith)

        # Frequency given by the caller
        if frequency is not None:
            freq = np.asarray(frequency, dtype=float)
            assert freq.shape == tec.shape  # Sanity
            return 5.308018e10 * tec / (4.0 * np.pi**2 * freq * freq)

        # Get frequency of the detected signal
        freq = np.zeros_like(obs.tstamps.jd)
        if obs.source.is_farfield:
            freq += obs.band.channels[0].sky_freq

        if obs.source.is_nearfield:

            # Get uplink and downlink TX epochs in TDB
            light_time = obs.tstamps.tdb - obs.tx_epochs.tdb  # type: ignore
            uplink_tx = obs.tx_epochs.tdb - light_time  # type: ignore
            downlink_tx = obs.tx_epochs.tdb  # type: ignore

            # Read three-way ramping data
            if obs.source.has_three_way_ramping:

                three_way = obs.source.three_way_ramping
                mask_3way = (
                    uplink_tx.jd[:, None] >= three_way["t0"].jd[None, :]  # type: ignore
                ) * (
                    uplink_tx.jd[:, None] <= three_way["t1"].jd[None, :]  # type: ignore
                )
                f0 = np.sum(np.where(mask_3way, three_way["f0"], 0), axis=1)
                df0 = np.sum(np.where(mask_3way, three_way["df"], 0), axis=1)
                t0 = np.sum(np.where(mask_3way, three_way["t0"].jd, 0), axis=1)
                dt = time.TimeDelta(uplink_tx.jd - t0, format="jd").to("s").value  # type: ignore
                freq += (f0 + df0 * dt) * self.three_way_turnaround_ratio()

                # Check for lack of coverage
                holes = np.sum(mask_3way, axis=1) == 0
            else:
                holes = np.ones_like(obs.tstamps.jd, dtype=int)

            # Fill holes in three-way ramping with one-way data
            if np.any(holes) and obs.source.has_one_way_ramping:

                one_way = obs.source.one_way_ramping
                mask_1way = (
                    (downlink_tx.jd[:, None] >= one_way["t0"].jd[None, :])  # type: ignore
                    * (downlink_tx.jd[:, None] <= one_way["t1"].jd[None, :])  # type: ignore
                    * holes[:, None]
                )
                f0 = np.sum(np.where(mask_1way, one_way["f0"], 0), axis=1)
                df0 = np.sum(np.where(mask_1way, one_way["df"], 0), axis=1)
                t0 = np.sum(np.where(mask_1way, one_way["t0"].jd, 0), axis=1)
                dt = (
                    time.TimeDelta(downlink_tx.jd - t0, format="jd")  # type: ignore
                    .to("s")
                    .value
                )
                freq += f0 + df0 * dt

                # Re-check for lack of coverage
                holes *= np.sum(mask_1way, axis=1) == 0

            # Fill remaining holes with constant frequency
            freq += np.where(holes, obs.source.default_frequency, 0)

        assert freq.shape == tec.shape  # Sanity

        return 5.308018e10 * tec / (4.0 * np.pi**2 * freq * freq)
