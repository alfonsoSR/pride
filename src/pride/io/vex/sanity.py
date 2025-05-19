from .interface import Vex
import datetime
from ...logger import log
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import ScanData


def get_zero_gaps_and_overlaps_between_scans(
    previous: "ScanData", current: "ScanData"
) -> tuple[list[str], list[str]]:
    """Identifies zero-gaps, and overlaps between two scans

    Given two data structures with scan data, checks for cases of zero-gap, and overlap for all their common stations.

    A zero-gap occurs when the start of the current scan (considering initial offset) is not separated from the end of the previous scan (considering final offset) by more than one second. An overlap occurs when the start of the current scan starts before the end of the previous one, also considering offsets.

    The function returns two lists, with the names of the stations for which each of these conditions is met.

    :param previous: Scan data for first scan
    :param current: Data for second scan
    :return zero_gap_cases: List of station IDs for zero-gap scans
    :return overlap_cases: List of station IDs for overlapping scans
    """

    overlap_cases: list[str] = []
    zero_gap_cases: list[str] = []

    # Check for overlaps for each station
    for station_id in previous.offsets_per_station:

        # Skip if station is not present in current scan
        if station_id not in current.offsets_per_station:
            continue

        # Get final epoch of previous scan for current station
        previous_offsets = previous.offsets_per_station[station_id]
        previous_end = previous.initial_epoch + datetime.timedelta(
            seconds=previous_offsets[1]
        )

        # Get initial epoch of current scan for current station
        current_offsets = current.offsets_per_station[station_id]
        current_start = current.initial_epoch + datetime.timedelta(
            seconds=(current_offsets[0])
        )

        # Calculate separation between scans
        diff = (current_start - previous_end).total_seconds()

        # If scans overlap, exit with error (diff < 0)
        if diff < 0:
            overlap_cases.append(station_id)
        elif not (diff > 1):
            zero_gap_cases.append(station_id)

    return zero_gap_cases, overlap_cases


def __get_list_of_zero_gap_scans_from_dictionary(
    scan_data_dictionary: dict[str, "ScanData"],
) -> list[tuple[str, str]]:
    """Encapsulates functionality of get_list_of_zero_gap_scans for testing"""

    # Initialize list of zero-gap cases
    zero_gap_cases: list[tuple[str, str]] = []

    # Initialize containers for previous scan data
    scan_ids = list(scan_data_dictionary.keys())
    previous_scan_id = scan_ids[0]
    previous_scan_data = scan_data_dictionary[previous_scan_id]

    # Iterate through the scans in the VEX
    for scan_id in scan_ids[1:]:

        # Get data for current scan
        scan_data = scan_data_dictionary[scan_id]

        # Get lists of zero-gap and overlap cases
        zero_gap_list, overlap_list = get_zero_gaps_and_overlaps_between_scans(
            previous_scan_data, scan_data
        )

        # If there are overlaps, exit with error
        if len(overlap_list) > 0:
            log.error(
                f"Scans {previous_scan_id} and {scan_id} overlap for "
                f"stations {overlap_list}"
            )
            exit(1)

        # Add zero-gap cases to the list
        for station_id in zero_gap_list:
            zero_gap_cases.append((scan_id, station_id))

        # Update previous scan data
        previous_scan_id = scan_id
        previous_scan_data = scan_data

    return zero_gap_cases


def get_list_of_zero_gap_scans(
    vex: "Vex", experiment_target: str
) -> list[tuple[str, str]]:
    """Identifies zero-gap scans in the VEX file

    A zero-gap scan is one that starts less than one second after the end of the previous one. A separation between scans of more than one second is required to prevent duplicated timestamps when calculating the numerical derivatives of rotation matrices. The function returns a list of scan-station pairs for which this condition is met.

    :param vex: Interface to the VEX file
    :param experiment_target: Name of the experiment target
    :return: List of tuples containing scan IDs and station IDs for zero-gap scans
    :raises SystemExit: When two scans containing a station overlap, meaning that one starts before the next one ends.
    """

    # Initialize containers for cases of zero gap
    zero_gap_cases: list[tuple[str, str]] = []

    # Load all the scans in the VEX into a dictionary
    scan_data_dictionary: dict[str, "ScanData"] = {
        scan_id: vex.load_single_scan_data(scan_id, experiment_target)
        for scan_id in vex.experiment_scans_ids
    }

    # Get list of zero-gap scans from the dictionary
    zero_gap_cases = __get_list_of_zero_gap_scans_from_dictionary(
        scan_data_dictionary
    )

    return zero_gap_cases
