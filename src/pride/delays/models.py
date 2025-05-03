from .core import Delay
from ..io import get_target_information, ESA_SPICE, NASA_SPICE, internal_file
from typing import TYPE_CHECKING, Any, Literal
from ..logger import log
import requests
from astropy import time, coordinates
from ftplib import FTP_TLS
import unlzw3
import gzip
import numpy as np
from scipy import interpolate

from ..types import Antenna
from ..external import vienna
from nastro import plots as ng
import spiceypy as spice
from ..experiment.source import FarFieldSource, NearFieldSource
from ..constants import J2000, L_C

if TYPE_CHECKING:
    from pathlib import Path
    from ..experiment.station import Station
    from ..experiment.observation import Observation


class Geometric(Delay):
    """Geometric delay"""

    name = "Geometric"
    etc = {}
    requires_spice = True
    station_specific = False

    def ensure_resources(self) -> None:

        # Find kernels for target
        target = get_target_information(self.exp.setup.general["target"])
        kernel_path: "Path" = self.config["data"] / target["short_name"]
        metak_path: "Path" = kernel_path / "metak.tm"
        kernel_source = ESA_SPICE if target["api"] == "ESA" else NASA_SPICE
        kernel_source += f"/{target['names'][-1]}/kernels"

        # Create kernel directory if not present
        if not kernel_path.exists():
            kernel_path.mkdir(parents=True)
            log.info(f"Created kernel directory for {target['names'][-1]}")

        # Download metakernel if not present
        if not metak_path.exists():
            log.info(f"Downloading metakernel for {target['names'][-1]}")
            metak_found: bool = False
            response = requests.get(
                f"{kernel_source}/mk/{target['meta_kernel'].upper()}"
            )
            if response.ok:
                metak_found = True
                metak_content = response.content.decode("utf-8").splitlines()
                with metak_path.open("w") as f:
                    for line in metak_content:
                        if "PATH_VALUES" in line:
                            line = line.replace("..", str(kernel_path))
                        f.write(f"{line}\n")
            if not metak_found:
                log.error(
                    "Failed to initialize geometric delay: "
                    f"Metakernel not found for {target['names'][-1]}"
                )
                exit(1)

        # Read list of kernels from metakernel
        klist: list[str] = []
        with metak_path.open() as metak:
            content = iter([line.strip() for line in metak.readlines()])
            for line in content:
                if "KERNELS_TO_LOAD" in line:
                    line = next(content)
                    while ")" not in line:
                        if len(line) > 0:
                            klist.append(line.replace("$KERNELS/", "")[1:-1])
                        line = next(content)
                    if line.split(")")[0] != "":
                        klist.append(
                            line.split(")")[0].replace("$KERNELS/", "")[1:-1]
                        )
                    break

        # Ensure that all kernels are present
        for kernel in klist:
            if not (kernel_path / kernel).exists():
                log.info(f"Downloading {kernel_source}/{kernel}")
                response = requests.get(f"{kernel_source}/{kernel}")
                if response.ok:
                    (kernel_path / kernel).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    (kernel_path / kernel).write_bytes(response.content)
                else:
                    log.error(
                        "Failed to initialize geometric delay: "
                        f"Failed to download {kernel_source}/{kernel}"
                    )
                    exit(1)

        return None

    def load_resources(self) -> dict[str, Any]:

        log.debug(f"Loading resources for {self.name} delay")

        return {}

    def calculate_nearfield(self, obs: "Observation") -> np.ndarray:
        """Calculate geometric delay for a near-field source

        NOTE: This function assumes that the phase center is the geocenter.
        """

        # log.warning(
        #     "Implementation of near-field geometric delay is not reliable"
        # )

        # Sanity
        source = obs.source
        assert isinstance(source, NearFieldSource)

        # Get TX epoch at spacecraft [Downlink SC -> Station]
        tx = obs.tx_epochs
        rx_station: time.Time = obs.tstamps.tdb  # type: ignore
        assert tx.scale == "tdb"  # Sanity
        assert rx_station.scale == "tdb"  # Sanity
        lt_station: np.ndarray = (rx_station - tx).to("s").value  # type: ignore

        # Calculate RX epoch at phase center [Geocenter]
        #######################################################################

        # Initialization
        clight = spice.clight() * 1e3
        clight2 = clight * clight

        # Calculate BCRF position of source at TX epoch
        et_tx: np.ndarray = (tx - J2000.tdb).to("s").value  # type: ignore
        xsrc_bcrf_tx = (
            np.array(
                spice.spkpos(source.spice_id, et_tx, "J2000", "NONE", "SSB")[0]
            )
            * 1e3
        )

        # Calculate BCRF position of phase center at estimated RX epoch
        # NOTE: Seems logical to initialize RX with the one of the station
        et_rx: np.ndarray = (rx_station - J2000.tdb).to("s").value  # type: ignore
        xphc_bcrf_rx = (
            np.array(spice.spkpos("EARTH", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )

        # Calculate gravitational parameter of celestial bodies
        _bodies: list = self.exp.setup.internal["lt_correction_bodies"]
        bodies = [bi for bi in _bodies if bi != "earth"]
        bodies_gm = (
            np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
            * 1e9
        )

        # Calculate BCRF positions of celestial bodies at TX
        xbodies_bcrf_tx = np.array(
            [
                np.array(spice.spkpos(body, et_tx, "J2000", "NONE", "SSB")[0])
                * 1e3
                for body in bodies
            ]
        )

        # Calculate position of source wrt celestial bodies at TX
        r0b = xsrc_bcrf_tx[None, :, :] - xbodies_bcrf_tx
        r0b_mag = np.linalg.norm(r0b, axis=-1)  # (M, N)

        # Initialize light travel time and rx epoch
        # lt_np1 = LT_{n+1}
        lt_np1 = np.linalg.norm(xphc_bcrf_rx - xsrc_bcrf_tx, axis=-1) / clight
        rx_np1: time.Time = tx + time.TimeDelta(lt_np1, format="sec")

        # Initialize variables for iteration
        lt_n = 0.0 * lt_np1
        n_iter = 0
        iter_max = self.exp.setup.internal["lt_max_iterations"]
        precision = float(self.exp.setup.internal["lt_precision"])

        # Iterative correction of LT and RX
        # Function: F(RX) = RX - TX - R_02/c - RLT_02
        # Derivative: dF/dRX = 1 - (R_02_vec * dR_2_vec/dRX) / (R_02 * c)
        # Newton-Raphson: RX_{n+1} = RX_n - F(RX_n) / dF/dRX
        # Equivalent: LT_{n+1} = LT_n - F(RX_n) / dF/dRX
        while np.any(np.abs(lt_np1 - lt_n) > precision) and n_iter < iter_max:

            # Update light travel time and RX
            lt_n = lt_np1
            rx_n = tx + time.TimeDelta(lt_n, format="sec")

            # Convert RX to ephemeris time
            et_rx = (rx_n - J2000.tdb).to("s").value  # type: ignore

            # Calculate BCRF coordinates of phase center at RX
            sphc_bcrf_rx = (
                np.array(
                    spice.spkezr("EARTH", et_rx, "J2000", "NONE", "SSB")[0]
                )
                * 1e3
            )
            xphc_bcrf_rx = sphc_bcrf_rx[:, :3]
            vphc_bcrf_rx = sphc_bcrf_rx[:, 3:]

            # Calculate BCRF positions of celestial bodies at RX
            xbodies_bcrf_rx = np.array(
                [
                    np.array(
                        spice.spkpos(body, et_rx, "J2000", "NONE", "SSB")[0]
                    )
                    * 1e3
                    for body in bodies
                ]
            )

            # Calculate relativistic correction
            r02 = xphc_bcrf_rx - xsrc_bcrf_tx  # (N, 3)
            r02_mag = np.linalg.norm(r02, axis=-1)  # (N,)
            r2b = xphc_bcrf_rx[None, :, :] - xbodies_bcrf_rx  # (M, N, 3)
            r2b_mag = np.linalg.norm(r2b, axis=-1)  # (M, N)
            r02b_mag = np.linalg.norm(r2b - r0b, axis=-1)  # (M, N)
            gmc = 2.0 * bodies_gm[:, None] / clight2  # (M, 1)
            rlt_02 = np.sum(
                (gmc / clight)
                * np.log(
                    (r0b_mag + r2b_mag + r02b_mag + gmc)
                    / (r0b_mag + r2b_mag - r02b_mag + gmc)
                ),
                axis=0,
            )

            # Evaluate function and derivative
            f = lt_n - (r02_mag / clight) - rlt_02
            dfdrx = (
                1.0
                - np.sum((r02 / r02_mag[:, None]) * vphc_bcrf_rx, axis=-1)
                / clight
            )

            # Update light travel time and RX
            lt_np1 = lt_n - f / dfdrx
            rx_np1 = tx + time.TimeDelta(lt_np1, format="sec")

            # Update iteration
            n_iter += 1

        # Calculate post-newtonian correction for path between stations
        # #####################################################################
        # NOTE: From now on, I refer to the RX of the station as rx1 and to the RX of the phase center as rx2

        # Calculate BCRF position of station at RX1
        xsta_gcrf_rx1 = obs.station.location(obs.tstamps, frame="icrf")
        rx1 = rx_station
        et_rx1 = (rx1 - J2000.tdb).to("s").value  # type: ignore
        searth_bcrf_rx1 = (
            np.array(spice.spkezr("EARTH", et_rx1, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        xearth_bcrf_rx1 = searth_bcrf_rx1[:, :3]
        vearth_bcrf_rx1 = searth_bcrf_rx1[:, 3:]
        xsun_bcrf_rx1 = (
            np.array(spice.spkpos("SUN", et_rx1, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        U_earth = (
            spice.bodvrd("SUN", "GM", 1)[1][0]
            * 1e9
            / np.linalg.norm(xsun_bcrf_rx1 - xearth_bcrf_rx1, axis=-1)
        )
        xsta_bcrf_rx1 = (
            xearth_bcrf_rx1
            + (1.0 - U_earth[:, None] / clight2 - L_C) * xsta_gcrf_rx1
            - 0.5
            * np.sum(vearth_bcrf_rx1 * xsta_gcrf_rx1, axis=-1)[:, None]
            * vearth_bcrf_rx1
            / clight2
        )

        # Calculate BCRF position of phase center at RX2
        rx2 = rx_np1
        et_rx2 = (rx2 - J2000.tdb).to("s").value  # type: ignore
        xphc_bcrf_rx2 = (
            np.array(spice.spkpos("EARTH", et_rx2, "J2000", "NONE", "SSB")[0])
            * 1e3
        )

        # Calculate position of celestial bodies at RX1 and RX2
        xbodies_bcrf_rx1 = (
            np.array(
                [
                    np.array(
                        spice.spkpos(body, et_rx1, "J2000", "NONE", "SSB")[0]
                    )
                    for body in bodies
                ]
            )
            * 1e3
        )
        xbodies_bcrf_rx2 = (
            np.array(
                [
                    np.array(
                        spice.spkpos(body, et_rx2, "J2000", "NONE", "SSB")[0]
                    )
                    for body in bodies
                ]
            )
            * 1e3
        )

        # Calculate relativistic correction
        r01 = xsta_bcrf_rx1 - xsrc_bcrf_tx  # (N, 3)
        r01_mag = np.linalg.norm(r01, axis=-1)  # (N,)
        r02 = xphc_bcrf_rx2 - xsrc_bcrf_tx  # (N, 3)
        r02_mag = np.linalg.norm(r02, axis=-1)  # (N,)
        r0b = xsrc_bcrf_tx[None, :, :] - xbodies_bcrf_tx  # (M, N, 3)
        r0b_mag = np.linalg.norm(r0b, axis=-1)
        r1b = xsta_bcrf_rx1[None, :, :] - xbodies_bcrf_rx1  # (M, N, 3)
        r1b_mag = np.linalg.norm(r1b, axis=-1)
        r2b = xphc_bcrf_rx2[None, :, :] - xbodies_bcrf_rx2  # (M, N, 3)
        r2b_mag = np.linalg.norm(r2b, axis=-1)
        gmc = 2.0 * bodies_gm[:, None] / (clight * clight)  # (M, 1)
        tg_12 = np.sum(
            (gmc / clight)
            * np.log(
                (r2b_mag + r0b_mag + r02_mag)
                * (r1b_mag + r0b_mag - r01_mag)
                / (
                    (r2b_mag + r0b_mag - r02_mag)
                    * (r1b_mag + r0b_mag + r01_mag)
                )
            ),
            axis=0,
        )

        # Calculate delay in TT
        #######################################################################
        dt: np.ndarray = (rx2 - rx1).to("s").value  # type: ignore
        vearth_mag = np.linalg.norm(vearth_bcrf_rx1, axis=-1)
        baseline = -xsta_gcrf_rx1
        v2 = 0.0 * vearth_bcrf_rx1  # Velocity of phase center in GCRF
        return -(
            (dt + tg_12)
            * (1 - (0.5 * vearth_mag * vearth_mag + U_earth) / clight2)
            / (1.0 - L_C)
            - np.sum(vearth_bcrf_rx1 * baseline, axis=-1) / clight2
        ) / (1.0 + np.sum(vearth_bcrf_rx1 * v2, axis=-1) / clight2)

    def calculate_farfield(self, obs: "Observation") -> np.ndarray:
        """Calculate geometric delay for a far-field source

        Calculates the geometric delay using the consensus model for far-field sources, described in section 11 of the IERS Conventions 2010.
        """

        # Sanity
        source = obs.source
        assert isinstance(source, FarFieldSource)

        # Initialization
        clight = spice.clight() * 1e3
        clight2 = clight * clight

        # Calculate baseline vector [For geocenter phase center it is just
        # the GCRF position of the station at RX]
        baseline = obs.station.location(obs.tstamps, frame="icrf")
        xsta_gcrf_rx = baseline

        # Calculate potential at geocenter
        et_rx: np.ndarray = (obs.tstamps.tdb - J2000.tdb).to("s").value  # type: ignore
        searth_bcrf_rx = (
            np.array(spice.spkezr("EARTH", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        xearth_bcrf_rx = searth_bcrf_rx[:, :3]
        vearth_bcrf_rx = searth_bcrf_rx[:, 3:]
        xsun_bcrf_rx = (
            np.array(spice.spkpos("SUN", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        gm_sun = spice.bodvrd("SUN", "GM", 1)[1][0] * 1e9
        U_earth = gm_sun / np.linalg.norm(
            xsun_bcrf_rx - xearth_bcrf_rx, axis=-1
        )

        # Calculate BCRF position of station at RX
        xsta_bcrf_rx = (
            xearth_bcrf_rx
            + (1.0 - U_earth[:, None] / clight2 - L_C) * xsta_gcrf_rx
            - 0.5
            * np.sum(vearth_bcrf_rx * xsta_gcrf_rx, axis=-1)[:, None]
            * vearth_bcrf_rx
            / clight2
        )

        # Position of phase center is just that of Earth for geocenter
        xphc_bcrf_rx = xearth_bcrf_rx

        # Calculate gravitational correction using IERS algorithm
        #######################################################################
        _bodies = self.exp.setup.internal["lt_correction_bodies"]
        bodies = [bi for bi in _bodies if bi != "earth"]
        bodies_gm = (
            np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
            * 1e9
        )
        xbodies_bcrf_rx = np.array(
            [
                np.array(spice.spkpos(body, et_rx, "J2000", "NONE", "SSB")[0])
                * 1e3
                for body in bodies
            ]
        )
        xbodies_sta_bcrf_rx = xbodies_bcrf_rx - xsta_bcrf_rx

        # Estimate time of closest approach to planets
        ks = obs.source.observed_ks  # Pointing vector
        et_planets = (
            et_rx
            - np.sum(ks[None, None, :] * xbodies_sta_bcrf_rx, axis=-1) / clight
        )
        et_closest = np.where(et_rx[None, :] < et_planets, et_rx, et_planets)

        # Calculate position of celestial bodies at closest approach
        xbodies_bcrf_closest = np.array(
            [
                np.array(spice.spkpos(body, et_body, "J2000", "NONE", "SSB")[0])
                * 1e3
                for body, et_body in zip(bodies, et_closest)
            ]
        )

        # Calculate correction
        r1j = xsta_bcrf_rx[None, :, :] - xbodies_bcrf_closest
        r1j_mag = np.linalg.norm(r1j, axis=-1)
        k_r1j = np.sum(ks[None, None, :] * r1j, axis=-1)
        r2j = (
            xphc_bcrf_rx[None, :, :]
            - np.sum(ks[None, :] * baseline, axis=-1)[None, :, None]
            * vearth_bcrf_rx[None, :, :]
            / clight
            - xbodies_bcrf_closest
        )
        r2j_mag = np.linalg.norm(r2j, axis=-1)
        k_r2j = np.sum(ks[None, :] * r2j, axis=-1)
        gmc = 2.0 * bodies_gm[:, None] / (clight * clight2)
        T_g = np.sum(
            gmc * np.log((r1j_mag + k_r1j) / (r2j_mag + k_r2j)), axis=0
        )

        # Calculate delay in TT
        ks_b = np.sum(ks[None, :] * baseline, axis=-1)
        ks_vearth = np.sum(ks[None, :] * vearth_bcrf_rx, axis=-1)
        vearth_b = np.sum(vearth_bcrf_rx * baseline, axis=-1)
        vearth_mag2 = np.linalg.norm(vearth_bcrf_rx, axis=-1) ** 2
        return (
            T_g
            - (ks_b / clight)
            * (1.0 - (2.0 * U_earth / clight2) - (0.5 * vearth_mag2 / clight2))
            - (vearth_b / clight2) * (1.0 + 0.5 * ks_vearth / clight)
        ) / (1.0 + ks_vearth / clight)

    def calculate(self, obs: "Observation") -> Any:

        if isinstance(obs.source, FarFieldSource):
            return self.calculate_farfield(obs)
        elif isinstance(obs.source, NearFieldSource):
            return self.calculate_nearfield(obs)
        else:
            log.error(
                "Failed to calculate geometric delay: Invalid source type"
            )
            exit(1)

        raise NotImplementedError("Missing calculate for geometric")


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
        date: time.Time = time.Time(
            self.exp.initial_epoch.mjd // 1, format="mjd", scale="utc"  # type: ignore
        )
        date.format = "iso"
        step = time.TimeDelta(1.0, format="jd")

        coverage: dict[tuple[time.Time, time.Time], Path] = {}
        while True:

            # Get gps week from date
            gps_week = int((date - self.etc["gps_week_ref"]).to("week").value)
            year = date.datetime.year  # type: ignore
            doy = date.datetime.timetuple().tm_yday  # type: ignore

            # Get file name and url for ionospheric data file
            if gps_week < self.etc["new_format_week"]:
                ionex_zip = f"igsg{doy:03d}0.{str(year)[2:]}i.Z"
            else:
                ionex_zip = (
                    f"IGS0OPS{self.etc['solution_type']}_{year:04d}{doy:03d}"
                    "0000_01D_02H_GIM.INX.gz"
                )
            ionex_file = self.config["data"] / ionex_zip
            # ionex_file = self.exp.setup.catalogues["ionospheric_data"] / ionex_zip
            url = f"{self.etc['url']}/{year:4d}/{doy:03d}/{ionex_zip}"

            # Ensure parent directory exists
            if not ionex_file.parent.exists():
                ionex_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file if not present
            if not ionex_file.with_suffix("").exists():

                if not ionex_file.exists():

                    log.info(f"Downloading {ionex_file.name}")

                    ftp = FTP_TLS(self.etc["ftp_server"])
                    ftp.login(user="anonymous", passwd="")
                    ftp.prot_p()
                    ftp.cwd(
                        "gps/products/ionex/" + "/".join(url.split("/")[-3:-1])
                    )
                    if not ionex_file.name in ftp.nlst():
                        raise FileNotFoundError(
                            "Failed to initialize ionospheric delay: "
                            f"Failed to download {url}"
                        )
                    ftp.retrbinary(
                        f"RETR {ionex_file.name}", ionex_file.open("wb").write
                    )

                # Uncompress file
                if ionex_file.suffix == ".Z":
                    ionex_file.with_suffix("").write_bytes(
                        unlzw3.unlzw(ionex_file.read_bytes())
                    )
                    ionex_file.unlink()
                elif ionex_file.suffix == ".gz":
                    with gzip.open(ionex_file, "rb") as f_in:
                        ionex_file.with_suffix("").write_bytes(f_in.read())
                    ionex_file.unlink()
                else:
                    raise ValueError(
                        "Failed to initialize ionospheric delay: "
                        "Invalid ionospheric data format"
                    )

            # Add coverage to dictionary
            coverage[(date, date + step)] = ionex_file.with_suffix("")

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
        _tec_maps: list[np.ndarray] = []

        for source in self.resources["coverage"].values():

            with source.open() as f:

                content = iter([line.strip() for line in f])
                lat_grid: np.ndarray | None = None
                lon_grid: np.ndarray | None = None
                ref_height: float = NotImplemented
                ref_rearth: float = NotImplemented

                while True:

                    try:
                        line = next(content)
                    except StopIteration:
                        break

                    # Read reference ionospheric height
                    if "HGT1 / HGT2 / DHGT" in line:
                        h1, h2 = np.array(line.split()[:2], dtype=float)
                        # Sanity check
                        if h1 != h2:
                            raise ValueError("Unexpected behavior")
                        ref_height = h1

                    # Read reference radius of the Earth
                    if "BASE RADIUS" in line:
                        ref_rearth = float(line.split()[0])

                    # Define latitude and longitude grids
                    if "LAT1 / LAT2 / DLAT" in line:
                        l0, l1, dl = np.array(line.split()[:3], dtype=float)
                        lat_grid = np.arange(l0, l1 + dl / 2, dl)
                    if "LON1 / LON2 / DLON" in line:
                        l0, l1, dl = np.array(line.split()[:3], dtype=float)
                        lon_grid = np.arange(l0, l1 + dl / 2, dl)

                    if "START OF TEC MAP" not in line:
                        continue
                    assert "START OF TEC MAP" in line
                    assert lat_grid is not None and lon_grid is not None
                    next(content)
                    next(content)

                    # Read TEC map
                    grid = np.zeros((len(lat_grid), len(lon_grid)))
                    for i, _ in enumerate(lat_grid):

                        grid[i] = np.array(
                            " ".join(
                                [next(content).strip() for _ in range(5)]
                            ).split(),
                            dtype=float,
                        )
                        line = next(content)

                    assert "END OF TEC MAP" in line
                    _tec_maps.append(grid)

        tec_maps = [
            interpolate.RegularGridInterpolator((lon_grid, lat_grid), grid.T)
            for grid in _tec_maps
        ]

        # Generate interpolators for the baselines
        resources: dict[str, Any] = {
            "ref_height": ref_height,
            "ref_rearth": ref_rearth,
        }
        for baseline in self.exp.baselines:

            coords = coordinates.EarthLocation(
                *baseline.station.location(tec_epochs, frame="itrf").T, unit="m"
            ).to_geodetic("GRS80")
            lat: np.ndarray = coords.lat.deg  # type: ignore
            lon: np.ndarray = coords.lon.deg  # type: ignore

            data = [
                tec_map([lon, lat])[0]
                for tec_map, lon, lat in zip(tec_maps, lon, lat)
            ]
            interp_type: str = "linear" if len(data) <= 3 else "cubic"
            resources[baseline.station.name] = interpolate.interp1d(
                tec_epochs.mjd, data, kind=interp_type
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


class AntennaDelays(Delay):
    """Delays due geometry and deformation of antennas

    # Available models
    ## Nothnagel
    - Source:  Nothnagel (2009) https://doi.org/10.1007/s00190-008-0284-z

    Required resources:
    - Temperature at station location (Obtained from site-specific Vienna)
    - Antenna information: Focus type, mount type, foundation height and thermal expansion coefficient and reference temperature
    """

    name = "AntennaDelays"
    etc = {
        "url": "https://vmf.geo.tuwien.ac.at/trop_products",
    }
    requires_spice = False
    station_specific = True

    def ensure_resources(self) -> None:

        # Initialize date from which to look for atmospheric data
        date: Any = time.Time(
            self.exp.initial_epoch.mjd // 1, format="mjd", scale="utc"  # type: ignore
        )
        step = time.TimeDelta(6 * 3600, format="sec")
        date += (self.exp.initial_epoch.datetime.hour // 6) * step  # type: ignore

        # Define coverage dictionary
        site_coverage: dict[tuple[time.Time, time.Time], Path] = {}

        # Acquire tropospheric data
        while True:

            # Get year and day of year
            year = date.datetime.year
            doy = date.datetime.timetuple().tm_yday

            # Site-specific data
            site_file = self.config["data"] / f"{year:04d}{doy:03d}.v3gr_r"
            site_url = (
                self.etc["url"]
                + f"/VLBI/V3GR/V3GR_OP/daily/{year:04d}/{site_file.name}"
            )
            if not site_file.parent.exists():
                site_file.parent.mkdir(parents=True, exist_ok=True)
            if not site_file.exists():
                log.info(f"Downloading {site_file}")
                response = requests.get(site_url)
                if not response.ok:
                    raise FileNotFoundError(
                        "Failed to initialize tropospheric delay: "
                        f"Failed to download {site_url}"
                    )
                site_file.write_bytes(response.content)

            if site_file not in site_coverage.values():
                site_coverage[(date, date + step)] = site_file
            else:
                key = list(site_coverage.keys())[
                    list(site_coverage.values()).index(site_file)
                ]
                site_coverage.pop(key)
                site_coverage[(key[0], date + step)] = site_file

            # Update date
            if date > self.exp.final_epoch:
                break
            date += step

        # Add coverage to resources
        self.resources["coverage"] = site_coverage

        return None

    def load_resources(self) -> dict[str, Any]:

        log.debug(f"Loading resources for {self.name} delay")

        resources: dict[str, tuple["Antenna", Any]] = {}

        for baseline in self.exp.baselines:

            station = baseline.station
            if station in resources:
                continue

            # Generate antenna object from thermal deformation data file
            _antenna = Antenna(ivs_name=station.name)
            with internal_file(
                self.exp.setup.catalogues["antenna_parameters"]
            ).open() as f:

                matching_antenna: str | None = None
                for line in f:
                    line = line.strip()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    if "ANTENNA_INFO" in line and any(
                        [x in line for x in station.possible_names]
                    ):
                        matching_antenna = line
                        break

                if matching_antenna is None:
                    log.warning(
                        f"Thermal deformation disabled for {station.name}: "
                        "Missing antenna parameters"
                    )
                    resources[station.name] = (_antenna, None)
                    continue
                else:
                    _antenna = Antenna.from_string(matching_antenna)

            # Load atmospheric data from site-specific Vienna files
            data = []
            for source in self.resources["coverage"].values():

                with source.open() as file:

                    content: str = ""
                    for line in file:
                        if np.any(
                            [name in line for name in station.possible_names]
                        ):
                            content = line
                            break

                    if content == "":
                        log.warning(
                            f"Thermal deformation disabled for {station.name}: "
                            "Missing atmospheric data"
                        )
                        break

                    _data = [float(x) for x in content.split()[1:]]
                    data.append([_data[0]] + _data[5:8])

            # Flag station if no data is available
            if len(data) == 0:
                resources[station.name] = (_antenna, None)
                continue

            data = np.array(data).T

            # Calculate humidity
            hum = self.humidity_model(data[2], data[3])

            interp_type: str = "linear" if len(data[0]) <= 3 else "cubic"
            _thermo = {
                "p": interpolate.interp1d(data[0], data[1], kind=interp_type),
                "TC": interpolate.interp1d(data[0], data[2], kind=interp_type),
                "hum": interpolate.interp1d(data[0], hum, kind=interp_type),
            }

            # Add station to resources
            resources[station.name] = (_antenna, _thermo)

        return resources

    def calculate(self, obs: "Observation") -> Any:
        """Groups thermal deformation and antenna axis offset"""

        dt_axis_offset = self.calculate_axis_offset(obs)
        dt_thermal_deformation = self.calculate_thermal_deformation(obs)
        return dt_axis_offset + dt_thermal_deformation

    def calculate_axis_offset(self, obs: "Observation") -> Any:

        # Load resources
        resources = self.loaded_resources[obs.station.name]
        antenna: Antenna = resources[0]
        if resources[1] is None:
            log.warning(f"{self.name} delay set to zero for {obs.station.name}")
            return np.zeros_like(obs.tstamps.jd)
        thermo: dict[str, Any] = resources[1]
        clight = spice.clight() * 1e3

        # Geodetic coordinates of station
        assert obs.tstamps.location is not None  # Sanity
        coords = obs.tstamps.location.to_geodetic("GRS80")
        lat: np.ndarray = np.array(coords.lat.rad, dtype=float)  # type: ignore
        lon: np.ndarray = np.array(coords.lon.rad, dtype=float)  # type: ignore

        # Determine unit vector along antenna fixed axis in VEN
        match antenna.mount_type:
            case "MO_AZEL":
                ax_uvec = np.array([1.0, 0.0, 0.0])[None, :]
            case "MO_EQUA":
                ax_uvec = np.array([np.sin(lat), 0.0 * lat, np.cos(lat)]).T
            case "MO_XYNO":
                ax_uvec = np.array([0.0, 0.0, 1.0])[None, :]
            case "MO_XYEA":
                ax_uvec = np.array([0.0, 1.0, 0.0])[None, :]
            case "MO_RICH":
                phi_0 = np.deg2rad(39.06)  # From Nothnagel (2009)
                delta_lambda = np.deg2rad(0.12)  # From Nothnagel (2009)
                ax_uvec = np.array(
                    [
                        np.sin(phi_0),
                        -np.cos(phi_0) * np.sin(delta_lambda),
                        np.cos(phi_0) * np.cos(delta_lambda),
                    ]
                )[None, :]
            case _:
                log.error(
                    f"Failed to calculate {self.name} delay for "
                    f"{antenna.ivs_name}: Invalid mount type"
                )
                exit(1)

        # Atmospheric conditions
        p = thermo["p"](obs.tstamps.mjd)
        p_hpa = p * 760.0 / 1013.25
        temp_k = thermo["TC"](obs.tstamps.mjd) + 273.16
        hum = thermo["hum"](obs.tstamps.mjd) / 100.0

        # Aberrated pointing vector corrected for atmospheric refraction
        rho = self.atmospheric_bending_angle(obs.source_el, temp_k, hum, p_hpa)
        zenith = 0.5 * np.pi - obs.source_el
        ks_vec = np.array(
            [
                np.cos(zenith - rho),
                np.sin(zenith - rho) * np.sin(obs.source_az),
                np.sin(zenith - rho) * np.cos(obs.source_az),
            ]
        ).T
        ks_uvec = ks_vec / np.linalg.norm(ks_vec, axis=-1)[:, None]

        # Axis offset vector in VEN
        ao_vec = np.cross(
            ax_uvec,
            np.cross(ks_uvec, ax_uvec, axis=-1),
            axis=-1,
        )
        ao_uvec = ao_vec / np.linalg.norm(ao_vec, axis=-1)[:, None]

        # Axis offset delay
        n_air = 77.6e-6 * p / temp_k + 1.0  # Refractive index of the air
        return -antenna.AO * np.sum(ks_uvec * ao_uvec, axis=-1) / clight * n_air

    @staticmethod
    def atmospheric_bending_angle(
        el: np.ndarray, temp: np.ndarray, hum: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        """Calculate atmospheric bending angle

        UNKNOWN MODEL - ORIGINAL CODE FROM DIMA'S PROGRAM

        :param el: Elevation of source [rad]
        :param temp: Temperature at station location [K]
        :param hum: Relative humidity at station location [%]
        :param p: Pressure at station location [mmHg = hPa]
        :return: Atmospheric bending angle [rad]
        """
        # log.debug("Missing source of atmospheric bending angle model")

        CDEGRAD = 0.017453292519943295
        CARCRAD = 4.84813681109536e-06

        a1 = 0.40816
        a2 = 112.30
        b1 = 0.12820
        b2 = 142.88
        c1 = 0.80000
        c2 = 99.344
        e = [
            46.625,
            45.375,
            4.1572,
            1.4468,
            0.25391,
            2.2716,
            -1.3465,
            -4.3877,
            3.1484,
            4.5201,
            -1.8982,
            0.89000,
        ]
        p1 = 760.0
        t1 = 273.0
        w = [22000.0, 17.149, 4684.1, 38.450]
        z1 = 91.870

        # Zenith angle in degrees
        z2 = np.rad2deg(0.5 * np.pi - el)
        # Temperature in Kelvin
        t2 = temp
        # Fractional humidity (0.0 -> 1.0)
        r = hum
        # Pressure in mm of Hg
        p2 = p

        # CALCULATE CORRECTIONS FOR PRES, TEMP, AND WETNESS
        d3 = 1.0 + (z2 - z1) * np.exp(c1 * (z2 - c2))
        fp = (p2 / p1) * (1.0 - (p2 - p1) * np.exp(a1 * (z2 - a2)) / d3)
        ft = (t1 / t2) * (1.0 - (t2 - t1) * np.exp(b1 * (z2 - b2)))
        fw = 1.0 + (
            w[0] * r * np.exp((w[1] * t2 - w[2]) / (t2 - w[3])) / (t2 * p2)
        )

        # CALCULATE OPTICAL REFRACTION
        u = (z2 - e[0]) / e[1]
        x = e[10]
        for i in range(8):
            x = e[9 - i] + u * x

        # COMBINE FACTORS AND FINISH OPTICAL FACTOR
        return (ft * fp * fw * (np.exp(x / d3) - e[11])) * CARCRAD

    def calculate_thermal_deformation(self, obs: "Observation") -> Any:
        """MODEL FROM DIMA'S CODE :: MISSING SOURCE"""

        # Load resources
        resources = self.loaded_resources[obs.station.name]
        antenna: Antenna = resources[0]
        thermo: dict[str, Any] = resources[1]
        if thermo is None:
            log.warning(f"{self.name} delay set to zero for {obs.station.name}")
            return np.zeros_like(obs.tstamps.jd)

        # Antenna focus factor based on focus type [See Nothnagel (2009)]
        match antenna.focus_type:
            case "FO_PRIM":
                focus_factor = 0.9
            case "FO_SECN":
                focus_factor = 1.8
            case _:
                log.error(
                    f"Failed to calculate {self.name} delay for "
                    f"{antenna.ivs_name}: Invalid focus type"
                )
                exit(1)

        # Interpolate atmospheric data at observation epochs
        temp = thermo["TC"](obs.tstamps.mjd)
        dT = temp - antenna.T0

        # Azimuth and elevation of source
        el: np.ndarray = obs.source_el
        ra: np.ndarray = obs.source_ra
        dec: np.ndarray = obs.source_dec

        # Calculate
        clight = spice.clight() * 1e3
        match antenna.mount_type:
            case "MO_AZEL":
                return (
                    antenna.gamma_hf * dT * antenna.hf * np.sin(el)
                    + antenna.gamma_hp
                    * dT
                    * (
                        antenna.hp * np.sin(el)
                        + antenna.AO * np.cos(el)
                        + antenna.hv
                        - focus_factor * antenna.hs
                    )
                ) / clight
            case "MO_EQUA":
                return (
                    antenna.gamma_hf * dT * antenna.hf * np.sin(el)
                    + antenna.gamma_hp
                    * dT
                    * (
                        antenna.hp * np.sin(el)
                        + antenna.AO * np.cos(dec)
                        + antenna.hv
                        - focus_factor * antenna.hs
                    )
                ) / clight
            case "MO_XYNO" | "MO_XYEA":
                print("using this one")
                return (
                    antenna.gamma_hf * dT * antenna.hf * np.sin(el)
                    + antenna.gamma_hp
                    * dT
                    * (
                        antenna.hp * np.sin(el)
                        + antenna.AO
                        * np.sqrt(
                            1.0
                            - np.cos(el) * np.cos(el) * np.cos(ra) * np.cos(ra)
                        )
                        + antenna.hv
                        - focus_factor * antenna.hs
                    )
                ) / clight
            case "MO_RICH":  # Misplaced equatorial (RICHMOND)

                # Error of the fixed axis and inclination wrt local horizon
                # Taken from Nothnagel (2009), for RICHMOND antenna
                phi_0 = np.deg2rad(39.06)
                delta_lambda = np.deg2rad(-0.12)

                return (
                    antenna.gamma_hf * dT * antenna.hf * np.sin(el)
                    + antenna.gamma_hp
                    * dT
                    * (
                        antenna.hp * np.sin(el)
                        + antenna.AO
                        * np.sqrt(
                            1.0
                            - (
                                np.sin(el) * np.sin(phi_0)
                                + np.cos(el)
                                * np.cos(phi_0)
                                * (
                                    np.cos(ra) * np.cos(delta_lambda)
                                    + np.sin(ra) * np.sin(delta_lambda)
                                )
                            )
                            ** 2
                        )
                        + antenna.hv
                        - focus_factor * antenna.hs
                    )
                ) / clight
            case _:
                log.error(
                    f"Failed to calculate {self.name} delay for "
                    f"{antenna.ivs_name}: Invalid mount type"
                )
                exit(1)

    @staticmethod
    def humidity_model(temp_c: np.ndarray, wvp: np.ndarray) -> np.ndarray:
        """Calculate relative humidity from temperature and water vapour pressure

        NOTE: Copy pasted from Dima. God knows where does this come from.
        """

        # Constants
        a = 10.79574
        c2k = 273.15
        b = 5.028
        c = 1.50475e-4
        d = 8.2969
        e = 0.42873e-3
        f = 4.76955
        g = 0.78614

        # Calculate saturation vapour pressure
        temp_k = temp_c + c2k
        ew = np.power(
            10.0,
            a * (1.0 - c2k / temp_k)
            - b * np.log10(temp_k / c2k)
            + c * (1 - np.power(10.0, d * (1.0 - temp_k / c2k)))
            + e * (np.power(10.0, f * (1.0 - temp_k / c2k)) - 1.0)
            + g,
        )

        # Calculate relative humidity
        return 100 * wvp / ew
