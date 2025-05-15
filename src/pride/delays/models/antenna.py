from ..core import Delay
from ... import io
from typing import TYPE_CHECKING, Any
from ...logger import log
import requests
from astropy import time
import numpy as np
from scipy import interpolate

from ...types import Antenna
import spiceypy as spice

if TYPE_CHECKING:
    from pathlib import Path
    from ...experiment.observation import Observation


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
            with io.internal_file(
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
