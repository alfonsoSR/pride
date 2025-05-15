from ..core import Displacement
from ...logger import log
from typing import Any
import numpy as np
from astropy import time
from ...external.iers import hardisp
from ... import io


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
