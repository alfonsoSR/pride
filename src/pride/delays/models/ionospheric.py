from ..core import Delay
from typing import TYPE_CHECKING, Any
from ...logger import log
from astropy import time, coordinates
from ftplib import FTP_TLS
import unlzw3
import gzip
import numpy as np
from scipy import interpolate
from ... import io, utils

if TYPE_CHECKING:
    from pathlib import Path
    from ...experiment.observation import Observation


class Ionospheric(Delay):
    """Ionospheric delay

    # Available models
    ## [MISSING NAME OF MODEL]
    - BRIEF DESCRIPTION OF THE MODEL AND SOURCE
    - DESCRIPTION OF REQUIRED RESOURCES AND KEYS
    """

    name = "Ionospheric"
    etc = {
        "url": "https://cddis.nasa.gov/archive/gps/products/ionex",
        "new_format_week": 2238,
        "gps_week_ref": time.Time("1980-01-06T00:00:00", scale="utc"),
        "model": "igs",
        "ftp_server": "gdc.cddis.eosdis.nasa.gov",
        "solution_type": "FIN",
    }
    requires_spice = False
    station_specific = True

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

        log.debug(f"Loading resources for {self.name} delay")

        # Generate TEC maps
        tec_epochs = time.Time([key[0] for key in self.resources["coverage"]])
        tec_grid_interpolators: list[interpolate.RegularGridInterpolator] = []

        # Reference values for Earth radius and ionospheric height
        reference_earth_radius: float = NotImplemented
        reference_ionospheric_height: float = NotImplemented

        # Load TEC maps from all the IONEX files
        for source in self.resources["coverage"].values():

            # Read IONEX file
            ionex_content = io.IonexInterface(source)
            tec_interpolators, ref_height, ref_rearth = (
                ionex_content.read_data_from_ionex_file()
            )

            # Update reference Earth radius or check consistency
            if reference_earth_radius is NotImplemented:
                reference_earth_radius = ref_rearth
            elif reference_earth_radius != ref_rearth:
                log.error(
                    f"Failed to load TEC maps: "
                    "Inconsistent reference Earth radius accross files"
                )
                exit(1)

            # Update reference ionospheric height or check consistency
            if reference_ionospheric_height is NotImplemented:
                reference_ionospheric_height = ref_height
            elif reference_ionospheric_height != ref_height:
                log.error(
                    f"Failed to load TEC maps: "
                    "Inconsistent reference ionospheric height accross files"
                )
                exit(1)

            # Update list of TEC map interpolators
            for tec_interpolator in tec_interpolators:
                tec_grid_interpolators.append(tec_interpolator)

        # Initialize resources dictionary with reference values
        resources: dict[str, Any] = {
            "ref_height": reference_ionospheric_height,
            "ref_rearth": reference_earth_radius,
        }

        # Generate a 1D TEC interpolator for each station
        for baseline in self.exp.baselines:

            # Station latitude and longitude for each coverage epoch
            coords = coordinates.EarthLocation(
                *baseline.station.location(tec_epochs, frame="itrf").T,
                unit="m",
            ).to_geodetic("GRS80")
            lat: np.ndarray = coords.lat.deg  # type: ignore
            lon: np.ndarray = coords.lon.deg  # type: ignore

            # Get TEC at station coordinates for each epoch in coverage
            tec_at_station_coordinates: list[float] = [
                float(tec_grid_interpolator([lon, lat])[0])
                for tec_grid_interpolator, lon, lat in zip(
                    tec_grid_interpolators, lon, lat
                )
            ]

            # Generate a 1D interpolator with the TEC at station coordinates
            # as function of the observation epoch
            interp_type: str = (
                "linear" if len(tec_at_station_coordinates) <= 3 else "cubic"
            )
            resources[baseline.station.name] = interpolate.interp1d(
                tec_epochs.mjd, tec_at_station_coordinates, kind=interp_type
            )

        return resources

    def calculate(self, obs: "Observation") -> Any:

        # Get vertical TEC from station at observation epochs
        vtec = self.loaded_resources[obs.station.name](obs.tstamps.mjd)

        # Read reference height and Earth radius from model
        h_ref = self.loaded_resources["ref_height"]
        r_earth = self.loaded_resources["ref_rearth"]

        # NOTE: Using model from Petrov (based on original code)
        zenith = np.arcsin(np.cos(obs.source_el) / (1.0 + (h_ref / r_earth)))
        tec = 0.1 * vtec / np.cos(zenith)

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
                freq += (f0 + df0 * dt) * obs.exp.setup.internal["tr_ratio"]

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
