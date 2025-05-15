from ..core import Delay
from typing import TYPE_CHECKING, Any
from ...logger import log
from astropy import time, coordinates
from ftplib import FTP_TLS
import unlzw3
import gzip
import numpy as np
from scipy import interpolate

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
