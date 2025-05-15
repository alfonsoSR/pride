from ..core import Displacement
from typing import Any
import numpy as np
from astropy import time
from ...constants import J2000


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
