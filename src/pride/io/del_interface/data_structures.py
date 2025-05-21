import numpy as np
from dataclasses import dataclass


@dataclass
class Scan:
    """Data structure for scan information

    :param scan_id: Scan ID (e.g. No0001)
    :param source: Source name
    :param mjd_ref: Integer component of the MJDs in the scan
    :param mjd2: Array of fractional components of the MJDs in the scan
    :u, v, w: Arrays of u, v, w projections [UNITS?]
    :param delays: Array of delays [s]
    :param doppler_phase: Array of Doppler phase values [UNITS?]
    :param doppler_amp: Array of Doppler amplitude values [UNITS?]
    """

    id: str
    source: str
    mjd_ref: int
    mjd2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    delays: np.ndarray
    doppler_phase: np.ndarray
    doppler_amp: np.ndarray


@dataclass
class DelContents:
    """Data structure for DEL file contents

    :param station_id: Station ID
    :param scans: Sequence of Scan objects
    """

    station_id: str
    scans: list[Scan]
