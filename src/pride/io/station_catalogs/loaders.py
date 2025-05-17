from ..resources import internal_file, load_catalog
import numpy as np
from pathlib import Path
from ...logger import log
from astropy import time

# Paths to relevant internal catalogs
INTERNAL_CATALOGS = load_catalog("config.yaml")["Catalogues"]
ALTERNATIVE_STATION_NAMES = load_catalog("station_names.yaml")
STATION_COORDINATES = internal_file(INTERNAL_CATALOGS["station_positions"])
STATION_VELOCITIES = internal_file(INTERNAL_CATALOGS["station_velocities"])


def load_reference_epoch_for_station_catalog() -> "time.Time":
    """Reference epoch of the station coordinates catalog

    :return reference_epoch: Reference epoch of the station coordinates catalog (UTC)
    """

    with STATION_COORDINATES.open("r") as file:

        # Initialize variable for line with reference epoch
        reference_epoch_string: str | None = None

        # Parse file
        for line in file:

            # Remove leading and trailing whitespace
            line = line.strip()

            # Check if line contains reference epoch
            if "EPOCH" in line:
                reference_epoch_string = line.split()[-1]
                break

    # If no matching line was found, raise an error
    if reference_epoch_string is None:
        log.error(
            f"Failed to load reference epoch for station coordinates: "
            f"Reference epoch not found in {STATION_COORDINATES}"
        )
        exit(1)

    # Convert to astropy Time object
    reference_epoch: "time.Time" = time.Time.strptime(
        reference_epoch_string, "%Y.%m.%d", scale="utc"
    )
    return reference_epoch


def load_station_coordinates_from_catalog(station_name: str) -> np.ndarray:
    """ITRF coordinates of a station at reference epoch of the catalog

    Parses the catalog of station coordinates, and returns the cartesian components of the ITRF position vector of the station at the reference epoch of the catalog. The function also uses the internal catalog of alternative station names in case the given name is not the one used in the catalog.

    :param station_name: Name of the station
    :return reference_position: ITRF coordinates of the station at reference epoch of the catalog
    """

    # Define list of possible names for the station
    possible_names: list[str] = [station_name]
    if station_name in ALTERNATIVE_STATION_NAMES:
        possible_names += ALTERNATIVE_STATION_NAMES[station_name]

    with STATION_COORDINATES.open("r") as file:

        # Initialize variable for line with station position
        matching_position: str | None = None

        # Parse file
        for line in file:

            # Remove leading and trailing whitespace
            line = line.strip()

            # Skip empty lines and comments
            if len(line) == 0 or line[0] == "$":
                continue

            # Check if any of the possible names is in the line
            if any([name in line for name in possible_names]):
                matching_position = line
                break

    # If no matching line was found, raise an error
    if matching_position is None:
        log.error(
            f"Failed to load coordinates for station {station_name}: "
            f"Station not found in {STATION_COORDINATES}"
        )
        exit(1)

    # Turn line into numpy array and return
    reference_position: np.ndarray = np.array(
        matching_position.split()[1:4], dtype=float
    )
    return reference_position


def load_station_velocity_from_catalog(station_name: str) -> np.ndarray:
    """ITRF velocity of a station at reference epoch of the catalog

    Parses the catalog of station velocities, and returns the cartesian components of the ITRF velocity vector of the station at the reference epoch of the catalog. The function also uses the internal catalog of alternative station names in case the given name is not the one used in the catalog.

    :param station_name: Name of the station
    :return reference_position: ITRF velocity of the station at reference epoch of the catalog
    """

    # Define list of possible names for the station
    possible_names: list[str] = [station_name]
    if station_name in ALTERNATIVE_STATION_NAMES:
        possible_names += ALTERNATIVE_STATION_NAMES[station_name]

    with STATION_VELOCITIES.open("r") as file:

        # Initialize variable for line with station velocity
        matching_velocity: str | None = None

        # Parse file
        for line in file:

            # Remove leading and trailing whitespace
            line = line.strip()

            # Skip empty lines and comments
            if len(line) == 0 or line[0] == "$":
                continue

            # Check if any of the possible names is in the line
            if any([name in line for name in possible_names]):
                matching_velocity = line
                break

    # If no matching line was found, raise an error
    if matching_velocity is None:
        log.error(
            f"Failed to load velocity for station {station_name}: "
            f"Station not found in {STATION_VELOCITIES}"
        )
        exit(1)

    # Turn line into numpy array and return
    reference_velocity: np.ndarray = (
        np.array(matching_velocity.split()[1:4], dtype=float) * 1e-3
    )
    return reference_velocity
