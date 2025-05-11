import numpy as np
from astropy import units
from . import io
import math
from .logger import log
import datetime


def eops_arcsec2rad(eops: np.ndarray) -> np.ndarray:
    """Convert EOPs from arcsec to rad

    :param eops: EOPs in s, arcsec
    :return: EOPs in s, rad
    """
    # Convert EOPs to s and rad [Original units are s and arcsec]
    ut1_utc, xp_as, yp_as, dx_as, dy_as = eops
    xp, yp, dx, dy = (
        units.Quantity([xp_as, yp_as, dx_as, dy_as], "arcsec").to("rad").value
    )
    return np.array([ut1_utc, xp, yp, dx, dy])


def discretize_scan(
    reference_epoch: datetime.datetime,
    initial_offset: int,
    final_offset: int,
    scan_id: str = "",
) -> list[datetime.datetime]:
    """Discretize a scan using internal constraints

    Given the reference epoch, and the initial and final offsets of a scan,
    discretizes it according to internal constraints defined in the `data`
    submodule (`config.yaml`). The constraints are:
    - Minimum number of observations per scan

    Given the initial and final offsets of a scan, the function calculates a
    step size required for the discretization to comply with the minimum number
    of observations per scan defined in `data/config.yaml`, as well as with the
    minimum allowed step size. If all the criteria are met, the function will
    return the default step size, and the associated number of observations.

    :param reference_epoch: Reference initial epoch of the scan
    :param initial_offset: Initial offset of the scan
    :param final_offset: Final offset of the scan
    :param scan_id: ID of the scan (for logging purposes)
    :return discretized_time_range: List of datetime objects representing the
    discretized time range for the scan
    """

    # Ensure that the offsets are positive and in order
    if initial_offset < 0 or final_offset < 0 or initial_offset > final_offset:
        log.error(
            f"Failed to discretize scan {scan_id}: "
            f"Invalid offsets {initial_offset}, {final_offset}"
        )
        exit(1)

    # Calculate initial epoch for the scan
    initial_epoch = reference_epoch + datetime.timedelta(seconds=initial_offset)

    # Load internal configuration for scan discretization
    internal_setup = io.load_catalog("config.yaml")["Configuration"]

    # Calculate the scan duration and a tentative step size
    scan_duration: int = final_offset - initial_offset
    min_extra_points: int = internal_setup["min_obs_per_scan"] - 1
    tentative_step: float = scan_duration / min_extra_points

    # Calculate number of extra points based on internal constraints
    # Number of observation is number of extra points + 1 (beginning)
    if tentative_step > internal_setup["default_scan_step"]:

        number_of_extra_points = math.ceil(
            scan_duration / internal_setup["default_scan_step"]
        )

    elif (
        internal_setup["min_scan_step"]
        <= tentative_step
        <= internal_setup["default_scan_step"]
    ):
        number_of_extra_points = math.ceil(scan_duration / tentative_step)

    else:
        number_of_extra_points = math.floor(
            scan_duration / internal_setup["min_scan_step"]
        )
        log.warning(f"Using minimum allowed step size for {scan_id}")

    # Recalculate the step size with correct number of extra points
    step_size = datetime.timedelta(
        seconds=scan_duration / number_of_extra_points
    )

    # Discretize the scan
    discretized_time_range: list[datetime.datetime] = [
        initial_epoch + step_size * i for i in range(number_of_extra_points + 1)
    ]

    return discretized_time_range
