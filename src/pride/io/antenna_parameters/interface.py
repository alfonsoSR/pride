from importlib import resources
from typing import Literal
from ... import utils
from ...logger import log
from dataclasses import dataclass

EXPECTED_PARAMETERS_PER_ANTENNA: int = 21


@dataclass(repr=False)
class AntennaParameters:
    """Data structure for antenna parameters

    Details in: Nothnagel (2009) https://doi.org/10.1007/s00190-008-0284-z

    Parameter descriptions from: antenna-info.txt (https://ivscc.gsfc.nasa.gov/IVS_AC/IVS-AC_data_information.htm)

    :param ivs_name: IVS station name
    :param focus_type: Focus type of the primary frequency
    :param mount_type: Mounting type
    :param radome: Whether the station has a radome
    :param meas_type: Measurement type (Complete, Incomplete, Rough)
    :param T0: Reference temperature [C]
    :param sin_T: Sine amplitude of annual temperature variations wrt J2000 epoch [C]
    :param cos_T: Cosine amplitude of annual temperature variations wrt J2000 epoch [C]
    :param h0: Reference pressure [hPa]
    :param ant_diam: Antenna diameter [m]
    :param hf: Height of the foundation [m]
    :param df: Depth of the foundation [m]
    :param gamma_hf: Thermal expansion coefficient of the foundation [1/K]
    :param hp: Length of the fixed axis [m]
    :param gamma_hp: Thermal expansion coefficient of the fixed axis [1/K]
    :param AO: Length of the offset between primary and secondary axes [m]
    :param gamma_AO: Thermal expansion coefficient of the offset [1/K]
    :param hv: Distance from the movable axis to the antenna vertex [m]
    :param gamma_hv: Thermal expansion coefficient of the structure connecting the movable axis to the antenna vertex [1/K]
    :param hs: Height of the subreflector/primary focus above the vertex [m]
    :param gamma_hs: Thermal expansion coefficient of the subreflector/primary focus mounting legs [1/K]
    """

    ivs_name: str
    focus_type: str
    mount_type: Literal[
        "MO_AZEL", "FO_PRIM", "MO_EQUA", "MO_XYNO", "MO_XYEA", "MO_RICH"
    ]
    radome: bool
    meas_type: Literal["ME_COMP", "ME_INCM", "ME_ROUG"]
    T0: float
    sin_T: float
    cos_T: float
    h0: float
    ant_diam: float
    hf: float
    df: float
    gamma_hf: float
    hp: float
    gamma_hp: float
    AO: float
    gamma_AO: float
    hv: float
    gamma_hv: float
    hs: float
    gamma_hs: float

    @staticmethod
    def from_catalog(station_name: str) -> "AntennaParameters":
        """Load antenna parameters from internal catalog

        Parses an internal catalog of antenna parameters (antenna-info.txt), reads the line corresponding to the specified station name, and packs the information into an AntennaParameters object.

        :param station_name: Name of the station
        :return: Data structure with antenna parameters
        """

        # Get line with parameters for this station from internal catalog
        raw_parameters = load_raw_parameters_from_catalog(station_name)

        # Raise error if station not found in catalog
        if raw_parameters is None:

            log.error(
                f"Failed to load antenna parameters for {station_name}: "
                "Station not found in internal catalog"
            )
            exit(1)

        # Turn line into tuple of parameters with correct types
        parameter_values = turn_raw_parameters_into_tuple(raw_parameters)

        return AntennaParameters(*parameter_values)  # type: ignore

    @staticmethod
    def from_catalog_missing_ok(
        station_name: str,
    ) -> "AntennaParameters | None":
        """Load antenna parameters from internal catalog

        Provides the same functionality as the `from_catalog` static method, but it returns None if the station is not found in the catalog instead of raising an error.

        :param station_name: Name of the station
        :return: Data structure with antenna parameters, or None if the station is not found in the catalog
        """

        # Get line with parameters for this station from internal catalog
        raw_parameters = load_raw_parameters_from_catalog(station_name)

        # Return with warning if station not found in catalog
        if raw_parameters is None:
            log.warning(
                f"Failed to load antenna parameters for {station_name}: "
                "Station not found in internal catalog"
            )
            return None

        # Turn line into tuple of parameters with correct types
        parameter_values = turn_raw_parameters_into_tuple(raw_parameters)
        return AntennaParameters(*parameter_values)  # type: ignore


def load_raw_parameters_from_catalog(
    station_name: str,
) -> str | None:
    """Get line with antenna parameters for a specific station

    Parses the internal catalog of antenna parameters and retrieves the line corresponding to the specified station name.

    :param station_name: Name of the station
    :return raw_parameters: Line with antenna parameters, or None if the station is not in the catalog
    """

    with resources.open_text("pride.data", "antenna-info.txt") as file:

        # Initialize variable for line with station data
        station_line: str | None = None

        # Parse contents of file
        for line in file:

            # Remove leading and trailing whitespace
            line = line.strip()

            # Skip empty lines and comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Skip lines that do not contain antenna information
            if "ANTENNA_INFO" not in line:
                continue

            # Check if line contains station data
            if utils.is_station_in_line(station_name, line):
                station_line = line
                break

    return station_line


def turn_raw_parameters_into_tuple(
    raw_parameters: str,
) -> tuple[str | bool | float, ...]:
    """Convert string of antenna parameters into a tuple of values

    :param raw_parameters: Line with antenna parameters from internal catalog
    :return: Tuple with antenna parameters that can be passed as input to the constructor of the data structure
    """

    # Split line into items and ensure correct format
    raw_parameter_list = raw_parameters.split()[1:]  # Skip "ANTENNA_INFO"
    if len(raw_parameter_list) != EXPECTED_PARAMETERS_PER_ANTENNA:
        log.error(
            "Failed to process raw antenna parameters for "
            f"{raw_parameter_list[0]}: "
            f"Expected {EXPECTED_PARAMETERS_PER_ANTENNA} parameters, "
            f"got {len(raw_parameter_list)}"
        )
        exit(1)

    # Extract relevant items into a tuple
    parameter_tuple = (
        *raw_parameter_list[:3],
        True if raw_parameter_list[3] == "RA_YES" else False,
        raw_parameter_list[4],
        *[float(x) for x in raw_parameter_list[5:]],
    )

    return parameter_tuple
