from ..core import Delay
from typing import TYPE_CHECKING, Any
from ...logger import log
import requests
from astropy import time
import numpy as np
from scipy import interpolate
from ... import utils, io
from ...external import vienna
import spiceypy as spice

if TYPE_CHECKING:
    from pathlib import Path
    from ...experiment.observation import Observation


class Tropospheric(Delay):
    """Tropospheric correction to light travel time"""

    name = "Tropospheric"
    etc = {
        "coords_url": "https://vmf.geo.tuwien.ac.at/station_coord_files/",
        "coeffs_url": (
            "https://vmf.geo.tuwien.ac.at/trop_products/VLBI/V3GR/"
            "V3GR_OP/daily/"
        ),
        "update_interval_hours": 6.0,  # Ignored, this was for Petrov
    }

    def ensure_resources(self) -> None:
        """Check for site coordinates and site-wise tropospheric data"""

        # Define range of dates to look for tropospheric data
        date = utils.get_date_from_epoch(self.exp.initial_epoch)
        step = time.TimeDelta(1.0, format="jd")

        # Download tropospheric data files
        coverage: dict[tuple[time.Time, time.Time], Path] = {}
        while True:

            # Download V3GR file for epoch
            site_file = io.download_v3gr_file_for_epoch(
                date, self.config["data"]
            )

            # Add coverage to dictionary
            coverage[(date, date + step)] = site_file

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage and source of coordinates to resources
        self.resources["coverage"] = coverage

        return None

    def load_resources(self) -> dict[str, dict[str, Any]]:

        log.info(f"Loading resources for {self.name} delay")

        resources: dict[str, dict[str, Any]] = {}
        for baseline in self.exp.baselines:

            # Check if resources are already available for this station
            station = baseline.station
            if station.name in resources:
                continue

            # Read site-wise tropospheric data
            _site_data = []
            for source in self.resources["coverage"].values():

                # Initialize V3GR interface for source file
                v3gr_interface = io.V3GRInterface(source)

                # Extract coefficients for station
                station_coefficients = (
                    v3gr_interface.read_v3gr_data_for_station(station.name)
                )
                _site_data.append(station_coefficients)

            data = np.array(_site_data).T

            # Interpolate coefficients
            mjd, ah, aw, dh, dw, gnh, geh, gnw, gew = data
            resources[station.name] = {
                "ah": interpolate.interp1d(mjd, ah, kind="linear"),
                "aw": interpolate.interp1d(mjd, aw, kind="linear"),
                "dh": interpolate.interp1d(mjd, dh, kind="linear"),
                "dw": interpolate.interp1d(mjd, dw, kind="linear"),
                "gnh": interpolate.interp1d(mjd, gnh, kind="linear"),
                "geh": interpolate.interp1d(mjd, geh, kind="linear"),
                "gnw": interpolate.interp1d(mjd, gnw, kind="linear"),
                "gew": interpolate.interp1d(mjd, gew, kind="linear"),
            }

        return resources

    def calculate(self, obs: "Observation") -> Any:

        # Initialization
        clight = spice.clight() * 1e3
        resources = self.loaded_resources[obs.station.name]
        mjd: np.ndarray = obs.tstamps.mjd  # type: ignore
        ah: np.ndarray = resources["ah"](mjd)
        aw: np.ndarray = resources["aw"](mjd)
        dh: np.ndarray = resources["dh"](mjd)
        dw: np.ndarray = resources["dw"](mjd)
        gnh: np.ndarray = resources["gnh"](mjd)
        geh: np.ndarray = resources["geh"](mjd)
        gnw: np.ndarray = resources["gnw"](mjd)
        gew: np.ndarray = resources["gew"](mjd)

        # Calculate geodetic coordinates of station
        assert obs.tstamps.location is not None
        coords = obs.tstamps.location.to_geodetic("GRS80")
        lat: np.ndarray = np.array(coords.lat.rad, dtype=float)  # type: ignore
        lon: np.ndarray = np.array(coords.lon.rad, dtype=float)  # type: ignore
        el: np.ndarray = obs.source_el
        az: np.ndarray = obs.source_az
        zd: np.ndarray = 0.5 * np.pi - el  # zenith distance

        # Calculate hydrostatic and wet mapping functions
        mfh, mfw = np.array(
            [
                vienna.vmf3(ah[i], aw[i], mjdi, lat[i], lon[i], zd[i])
                for i, mjdi in enumerate(mjd)
            ]
        ).T

        # Calculate gradient mapping function
        # SOURCE: https://doi.org/10.1007/s00190-018-1127-1
        sintan = np.sin(el) * np.tan(el)
        mgh = 1.0 / (sintan + 0.0031)
        mgw = 1.0 / (sintan + 0.0007)

        # Calculate delay
        return (
            dh * mfh
            + dw * mfw
            + mgh * (gnh * np.cos(az) + geh * np.sin(az))
            + mgw * (gnw * np.cos(az) + gew * np.sin(az))
        ) / clight
