from ..core import Displacement
from typing import Any
import numpy as np
from astropy import time
import spiceypy as spice
from ...constants import J2000
from ...external.iers import dehanttideinel


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
