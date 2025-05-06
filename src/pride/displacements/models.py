from .core import Displacement
from ..logger import log
from typing import Any, TYPE_CHECKING
import numpy as np
from astropy import time
import spiceypy as spice
from ..constants import J2000
from ..external.iers import dehanttideinel, hardisp
from .. import io


class SolidTide(Displacement):
    """Displacement due to solid Earth tides

    Implements the conventional model for displacements due to solid Earth tides induced by the Sun and the Moon as described in section 7.1.1 of the IERS Conventions 2010.
    """

    name: str = "SolidTide"
    requires_spice: bool = True

    def ensure_resources(self) -> None:
        return None

    def load_resources(
        self, epoch: "time.Time", shared: dict[str, Any]
    ) -> dict[str, Any]:

        # Position of the Sun and Moon in Earth-centered ICRF
        et: np.ndarray = (epoch.tdb - J2000.tdb).sec  # type: ignore
        xsun_icrf = (
            np.array(spice.spkpos("sun", et, "J2000", "NONE", "earth")[0]) * 1e3
        )
        xmoon_icrf = (
            np.array(spice.spkpos("moon", et, "J2000", "NONE", "earth")[0])
            * 1e3
        )

        # Convert position of the Sun and Moon to ITRF
        resources = {
            "xsta_itrf": shared["xsta_itrf"],
            "xsun_itrf": (shared["icrf2itrf"] @ xsun_icrf[:, :, None])[:, :, 0],
            "xmoon_itrf": (shared["icrf2itrf"] @ xmoon_icrf[:, :, None])[
                :, :, 0
            ],
        }

        return resources

    def calculate(
        self, epoch: "time.Time", resources: dict[str, Any]
    ) -> np.ndarray:

        # State vectors of station, Sun and Moon in ITRF
        xsta_itrf = resources["xsta_itrf"]
        xsun_itrf = resources["xsun_itrf"]
        xmoon_itrf = resources["xmoon_itrf"]

        out: np.ndarray = np.zeros_like(xsta_itrf)
        for idx, (ti, xsta, xsun, xmoon) in enumerate(
            zip(epoch, xsta_itrf, xsun_itrf, xmoon_itrf)
        ):
            yr, month, day, hour, min, sec = ti.datetime.timetuple()[:6]  # type: ignore
            fhr = hour + min / 60.0 + sec / 3600.0
            out[idx] = dehanttideinel(xsta, yr, month, day, fhr, xsun, xmoon)

        return out


class OceanLoading(Displacement):
    """Displacement due to ocean loading

    Implements the conventional model for displacements due to ocean loading as described in section 7.1.2 of the IERS Conventions 2010.
    """

    name: str = "OceanLoading"
    requires_spice: bool = False
    model: str = "tpxo72"

    def ensure_resources(self) -> None:

        source = io.internal_file(f"{self.model}.blq")
        if not source.exists():
            log.error(
                f"Failed to initialize {self.name} displacement: {source} not found"
            )
            log.info(
                "Downloading ocean loading data will be supported in the future"
            )
            exit(1)

        self._resources["source"] = source

        return None

    def load_resources(
        self, epoch: time.Time, shared: dict[str, Any]
    ) -> dict[str, Any]:

        with self._resources["source"].open("r") as f:

            content = f.readlines()

            _amp: np.ndarray | None = None
            _phs: np.ndarray | None = None

            for idx, line in enumerate(content):
                line = line.strip()
                if len(line) == 0 or line[0] == "$":
                    continue
                if any([name in line for name in shared["station_names"]]):
                    idx += 1
                    while content[idx][0] == "$":
                        idx += 1
                    line = content[idx].strip()
                    _amp = np.array(
                        " ".join(content[idx : idx + 3]).split(),
                        dtype=float,
                    ).reshape((3, 11))
                    _phs = np.array(
                        " ".join(content[idx + 3 : idx + 6]).split(),
                        dtype=float,
                    ).reshape((3, 11))
                    break

        if _amp is None or _phs is None:
            log.error(
                f"Failed to load ocean loading data for {shared['station']}"
            )
            exit(1)

        resources = {
            "amp": _amp,
            "phs": _phs,
            "seu2itrf": shared["seu2itrf"],
        }

        return resources

    def calculate(
        self, epoch: "time.Time", resources: dict[str, Any]
    ) -> np.ndarray:

        # Calculate ocean loading displacements
        dv, dw, ds = np.zeros((3, len(epoch)))
        for idx, ti in enumerate(epoch):
            dv[idx], dw[idx], ds[idx] = hardisp.hardisp(
                str(ti.isot)[:-4],  # type: ignore
                resources["amp"],
                resources["phs"],
                1,
                1,
            )

        # Convert displacements to ITRF
        disp_seu = np.array([ds, -dw, dv])
        out: np.ndarray = (
            resources["seu2itrf"] @ disp_seu.T[:, :, None]
        ).squeeze()
        return out


class PoleTide(Displacement):
    """Rotational deformation due to pole tide

    Implements the conventional model for rotational deformation due to pole tide as described in section 7.1.4 of the IERS Conventions 2010.
    """

    name: str = "PoleTide"
    requires_spice: bool = False

    def ensure_resources(self) -> None:
        return None

    def load_resources(
        self, epoch: "time.Time", shared: dict[str, Any]
    ) -> dict[str, Any]:

        resources: dict[str, Any] = {
            "eops": shared["eops"],
            "old_model": np.array(
                [
                    [55.974, 1.8243, 0.18413, 0.007024],
                    [346.346, 1.7896, -0.10729, -0.000908],
                ]
            )
            * 1e-3,
            "new_model": np.array(
                [[23.513, 7.6141, 0.0, 0.0], [358.891, -0.6287, 0.0, 0.0]]
            )
            * 1e-3,
            "lat": shared["lat"],
            "lon": shared["lon"],
            "seu2itrf": shared["seu2itrf"],
        }

        return resources

    def calculate(
        self, epoch: "time.Time", resources: dict[str, Any]
    ) -> np.ndarray:

        # Select IERS model based on epoch
        dt: np.ndarray = (epoch - J2000).to("year").value  # type: ignore
        use_old = dt < 10.0
        model = (
            use_old[:, None, None] * resources["old_model"][None, :, :]
            + (1.0 - use_old[:, None, None])
            * resources["new_model"][None, :, :]
        )

        # Calculate m1 and m2
        pow_dt = np.pow(dt[:, None], np.arange(4)[None, :])
        p_mean = (model @ pow_dt[:, :, None])[:, :, 0]
        m1, m2 = (resources["eops"][1:3] - p_mean.T) * np.array([[1.0], [-1.0]])

        # Calculate pole tide displacements in SEU system
        lat, lon = resources["lat"], resources["lon"]
        cospsin = m1 * np.cos(lon) + m2 * np.sin(lon)
        sinmcos = m1 * np.sin(lon) - m2 * np.cos(lon)
        disp_seu = np.array(
            [
                -9e-3 * np.cos(2.0 * lat) * cospsin,
                9e-3 * np.cos(lat) * sinmcos,
                -33e-3 * np.sin(2.0 * lat) * cospsin,
            ]
        ).T

        # Convert displacements to ITRF
        out: np.ndarray = (
            resources["seu2itrf"] @ disp_seu[:, :, None]
        ).squeeze()
        return out
