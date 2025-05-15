from ..core import Delay
from typing import TYPE_CHECKING, Any
from ...logger import log
import requests
from astropy import time
import numpy as np
from scipy import interpolate

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
        "update_interval_hours": 6.0,
    }

    def ensure_resources(self) -> None:
        """Check for site coordinates and site-wise tropospheric data"""

        # # Download source of site coordinates if not present
        # vlbi_ell = self.config["data"] / "vlbi.ell"
        # vlbi_ell_url = self.etc["coords_url"] + vlbi_ell.name
        # if not vlbi_ell.exists():
        #     response = requests.get(vlbi_ell_url)
        #     if not response.ok:
        #         log.error(
        #             "Failed to initialize tropospheric delay: "
        #             f"Failed to download {vlbi_ell_url}"
        #         )
        #         exit(1)
        #     vlbi_ell.write_bytes(response.content)

        # Initialize date from which to look for tropospheric data
        date: time.Time = time.Time(
            self.exp.initial_epoch.mjd // 1,  # type: ignore
            format="mjd",
            scale="utc",
        )
        step = time.TimeDelta(
            self.etc["update_interval_hours"] * 3600.0, format="sec"
        )
        date += (
            self.exp.initial_epoch.datetime.hour  # type: ignore
            // self.etc["update_interval_hours"]
        ) * step

        # Download tropospheric data files
        coverage: dict[tuple[time.Time, time.Time], Path] = {}
        while True:

            # Define filename and url
            year: Any = date.datetime.year  # type: ignore
            doy: Any = date.datetime.timetuple().tm_yday  # type: ignore
            site_file = self.config["data"] / f"{year:04d}{doy:03d}.v3gr_r"
            site_url = self.etc["coeffs_url"] + f"{year:04d}/{site_file.name}"

            # Download file if not present
            site_file.parent.mkdir(parents=True, exist_ok=True)
            if not site_file.exists():
                log.info(f"Downloading {site_file}")
                response = requests.get(site_url)
                if not response.ok:
                    log.error(
                        "Failed to initialize tropospheric delay: "
                        f"Failed to download {site_url}"
                    )
                    exit(1)
                site_file.write_bytes(response.content)

            # Add to coverage if necessary
            if site_file not in coverage.values():
                coverage[(date, date + step)] = site_file
            else:
                key = list(coverage.keys())[
                    list(coverage.values()).index(site_file)
                ]
                coverage.pop(key)
                coverage[(key[0], date + step)] = site_file

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage and source of coordinates to resources
        self.resources["coverage"] = coverage

        return None

    def load_resources(self) -> dict[str, dict[str, Any]]:

        resources: dict[str, dict[str, Any]] = {}
        for baseline in self.exp.baselines:

            # Check if resources are already available for this station
            station = baseline.station
            if station.name in resources:
                continue

            # # Read site coordinates
            # with self.resources["sites"].open() as f:

            #     # Find site in file
            #     _content: str = ""
            #     for line in f:
            #         if np.any(
            #             [name in line for name in station.possible_names]
            #         ):
            #             _content = line
            #             break

            #     # Ensure that site is present
            #     if _content == "":
            #         log.error(
            #             "Failed to initialize tropospheric delay: "
            #             f"Site-wise data not available for {station.name}"
            #         )
            #         exit(1)

            # lat, lon, height = [float(x) for x in _content.split()[1:4]]
            # site_coords = coordinates.EarthLocation.from_geodetic(
            #     lat=lat, lon=lon, height=height, ellipsoid="GRS80"
            # )

            # Read site-wise tropospheric data
            _site_data = []
            for source in self.resources["coverage"].values():

                # Find site in file
                with source.open() as f:
                    _content: str = ""
                    for line in f:
                        if np.any(
                            [name in line for name in station.possible_names]
                        ):
                            _content = line
                            break

                # Ensure that site is present
                if _content == "":
                    log.error(
                        "Failed to initialize tropospheric delay: "
                        f"Site-wise data not available for {station.name}"
                    )
                    exit(1)

                # Extract coefficients
                content = _content.split()
                _site_data.append(
                    [float(x) for x in content[1:6]]
                    + [float(x) for x in content[9:]]
                )

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
