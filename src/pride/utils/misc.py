import numpy as np
from astropy import units
import datetime
from ..logger import log
from .. import io
import math
from typing import Sequence, Callable, Any
from scipy import interpolate
from importlib import resources
import yaml
from pathlib import Path
import struct

# Load internal configuration
with resources.path("pride.data", "config.yaml") as config_path:

    __config = yaml.safe_load(config_path.open())
    INTERNAL_CATALOGS: dict[str, str] = __config["Catalogues"]
    INTERNAL_CONFIGURATION: dict[str, Any] = __config["Configuration"]
    ALTERNATIVE_STATION_NAMES: dict[str, list[str]] = yaml.safe_load(
        (
            config_path.parent / INTERNAL_CATALOGS["alternative_station_names"]
        ).open()
    )


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

    # Calculate the scan duration and a tentative step size
    scan_duration: int = final_offset - initial_offset
    min_extra_points: int = INTERNAL_CONFIGURATION["min_obs_per_scan"] - 1
    tentative_step: float = scan_duration / min_extra_points

    # Calculate number of extra points based on internal constraints
    # Number of observation is number of extra points + 1 (beginning)
    if tentative_step > INTERNAL_CONFIGURATION["default_scan_step"]:

        number_of_extra_points = math.ceil(
            scan_duration / INTERNAL_CONFIGURATION["default_scan_step"]
        )

    elif (
        INTERNAL_CONFIGURATION["min_scan_step"]
        <= tentative_step
        <= INTERNAL_CONFIGURATION["default_scan_step"]
    ):
        number_of_extra_points = math.ceil(scan_duration / tentative_step)

    else:
        number_of_extra_points = math.floor(
            scan_duration / INTERNAL_CONFIGURATION["min_scan_step"]
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


def is_station_in_line(station_name: str, line: str) -> bool:
    """Check if a line of a data file corresponds to a station

    Since station names are not unique, checking if a line of a data file contains information about a station requires certain logic. This function takes a station name, retrieves a list of possible names from an internal catalog, and checks if any of them is present in the line.

    The function splits the line into words to prevent returning True when the station name is part of some word. For example, if station name is HART, the function should return False for a line containing HARTAO

    :param station_name: Name of the station as specified in the internal catalog
    :param line: Line of the data file
    :return: True if the line corresponds to the station, False otherwise
    """

    # Get alternative names for the station
    alternative_names: list[str] = [station_name]
    if station_name in ALTERNATIVE_STATION_NAMES:
        alternative_names += ALTERNATIVE_STATION_NAMES[station_name]

    # Check if any of the alternative names is present in the line
    return any([name in line.split() for name in alternative_names])


def peek_buffer(
    buffer: bytes, contents_format: str, start: int
) -> tuple[list[Any], int]:
    """Read and decode a sequence of bytes from a buffer

    Starting from the `start` position, reads a portion of the buffer, and decodes it according to the format string. The amount of bytes to read is calculated automatically from the format string. The function returns the decoded contents, and the updated position from which to keep reading the buffer.

    :param buffer: Buffer of bytes to read from
    :param contents_format: Format string to decode the bytes
    :param start: Position from which to start reading the buffer
    :return contents: Decoded contents
    :return current_byte: Updated position from which to keep reading the buffer
    :raises BufferError: If it is not possible to calculate the number of bytes to read from the format string
    :raises BufferError: If the sum of the start position, and the calculated number of bytes to read is greater than the size of the buffer
    """

    # Get number of bytes to read from requested format
    try:
        bytes_to_read: int = struct.calcsize(contents_format)
    except:
        raise BufferError(
            f"Failed to calculate size of format string: {contents_format}"
        )

    # Avoid trying to read beyond the end of the buffer
    if start + bytes_to_read > len(buffer):
        raise BufferError("Requested to read past the end of the buffer")

    # Unpack contents of buffer
    contents = struct.unpack(
        contents_format, buffer[start : start + bytes_to_read]
    )

    # Post-process output to turn bytes into strings
    postprocessed_contents: list[str | float | int] = []
    for item in contents:

        # If integer or float, append it to the list
        if not isinstance(item, bytes):
            postprocessed_contents.append(item)
            continue

        # Decode bytes to string
        decoded_item: str = item.decode("utf-8")

        # Remove padding and white-space
        decoded_item = decoded_item.replace("\x00", "").replace(" ", "")

        # Append decoded item to list
        postprocessed_contents.append(decoded_item)

    # Return contents and updated position
    return postprocessed_contents, start + bytes_to_read
