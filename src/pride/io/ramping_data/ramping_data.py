from pathlib import Path
from ...logger import log
from astropy import time
import numpy as np
from typing import Literal


def __data_is_in_time_interval(
    data_t0: str,
    data_tend: str,
    time_interval: tuple[time.Time, time.Time] | None,
) -> tuple[bool, bool]:
    """Check if a line falls within the desired time interval

    :param data_t0: Beginning of coverage for the data in the line
    :param data_tend: End of coverage for the data in the line
    :param time_interval: Beginning and end of interval of interest
    :return line_in_interval: Whether the data in the line is valid for the requested interval
    :return is_after_interval: Whether the line contains data for moments that
    are after the end of the interval. This is used to stop reading the file
    when the rest of the data it contains is no longer relevant.
    """

    # If no time interval, data is valid
    if time_interval is None:
        return True, False

    # Convert limits of coverage to astropy Time objects
    t0: time.Time = time.Time(data_t0)
    tend: time.Time = time.Time(data_tend)

    # Check if line coverage ends before start of the interval
    if tend < time_interval[0]:
        return False, False

    # Check if line coverage starts after the end of the interval
    if t0 > time_interval[-1]:
        return False, True

    # Otherwise, at least part of the data is contained in the interval
    return True, False


def load_ramping_data(
    ramping_file: Path,
    ramping_type: Literal["one-way", "three-way"],
    time_interval: tuple[time.Time, time.Time] | None = None,
) -> dict[str, time.Time | np.ndarray | str] | None:
    """Load frequency ramping data from source file

    Parses a file containing frequency ramping data, and loads the part of its
    contents that falls within a specified range of dates into a dictionary.

    :param ramping_file: Path to the source file for ramping data
    :param ramping_type: Whether the file contains one-way or three-way data.
    :param time_interval: Time interval from which data is wanted. If not set,
    the whole content of the source file will be loaded.
    :return ramping_data: A dictionary containing the ramping data between
    specified epochs. For one-way ramping, the dictionary contains a Time object
    with the beginning of coverage for each line (t0), a Time object with the
    end of coverage for each line (t1), an array with reference frequencies (f0),
    and an array with derivatives of frequency during coverage (df). For
    three-way ramping, it contains an additional entry with the name of the
    uplink station for each line (uplink).
    """

    # If source file does not exist, return with warning
    ramping_file = ramping_file.absolute()
    if not ramping_file.exists():
        log.warning(
            "Failed to load ramping data: " f"File {ramping_file} not found."
        )
        return None

    # If ramping type is not valid, return with error
    if ramping_type not in ("one-way", "three-way"):
        log.error(
            f"Failed to load ramping data: Invalid ramping type {ramping_type}"
        )
        exit(1)

    # Initialize data containers
    coverage_start: list[str] = []
    coverage_end: list[str] = []
    reference_frequencies: list[float] = []
    frequency_ramping: list[float] = []
    uplink_station_names: list[str] = []

    # Read data from file into containers
    with ramping_file.open("r") as f:
        for line in f:

            # Skip comments and empty lines
            line_content: list[str] = line.strip().split()
            if "#" in line or len(line_content) == 0:
                continue

            # Check if data falls within the requested interval
            line_t0_string: str = "T".join(line_content[:2])
            line_tend_string: str = "T".join(line_content[2:4])
            line_in_interval, should_stop_reading = __data_is_in_time_interval(
                line_t0_string, line_tend_string, time_interval
            )
            if not line_in_interval:
                if should_stop_reading:
                    break
                else:
                    continue

            # Add data to lists
            coverage_start.append(line_t0_string)
            coverage_end.append(line_tend_string)
            reference_frequencies.append(float(line_content[4]))
            frequency_ramping.append(float(line_content[5]))
            if ramping_type == "three-way":
                uplink_station_names.append(line_content[6])

    # If lists are empty, return None
    if len(coverage_start) == 0:
        return None

    # Turn lists into dictionary
    ramping_data: dict = {
        "t0": time.Time(coverage_start),
        "t1": time.Time(coverage_end),
        "f0": np.array(reference_frequencies),
        "df": np.array(frequency_ramping),
    }
    if ramping_type == "three-way":
        ramping_data["uplink"] = uplink_station_names

    return ramping_data
