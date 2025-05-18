from pathlib import Path
from ...logger import log
import numpy as np
from typing import Iterator
from scipy import interpolate
from ... import utils


class V3GRInterface:
    """Interface to content of V3GR files"""

    def __init__(self, v3gr_file: Path) -> None:
        """Constructor for V3GRInterface class

        :param v3gr_file: Path to the V3GR file
        """

        # Ensure the file exists
        self.v3gr_file = v3gr_file
        if not self.v3gr_file.exists():
            log.error(
                f"Failed to parse V3GR file {self.v3gr_file}: "
                "File does not exist"
            )
            exit(1)

        return None

    def __find_station_line_in_v3gr_file(self, station_name: str) -> str:
        """Find the line in the V3GR file that contains the station data

        :param station_name: Name of the station
        :return: Line containing the station data
        """

        with self.v3gr_file.open() as file:

            # Initialize variable for line with station data
            station_line: str | None = None

            # Parse contents of file
            for line in file:

                # Remove leading and trailing whitespace
                line = line.strip()  # Probably not needed

                # Check if line contains station data
                if utils.is_station_in_line(station_name, line):
                    station_line = line
                    break

        # Ensure that station data was found
        if station_line is None:
            log.error(
                f"Failed load V3GR data for {station_name}: "
                f"Station not found in {self.v3gr_file}"
            )
            exit(1)

        return station_line

    def read_v3gr_data_for_station(self, station_name: str) -> list[float]:
        """Read V3GR data for a specific station

        :param station_name: Name of the station
        :return site_data: List of coefficients for the station
        """

        # Find the line in the V3GR file that contains the station data
        station_line = self.__find_station_line_in_v3gr_file(station_name)

        # Extract MJD and coefficients from line
        line_contents = station_line.split()
        coefficients = [float(x) for x in line_contents[1:6]] + [
            float(x) for x in line_contents[9:]
        ]
        return coefficients

    def read_atmospheric_conditions_at_station(
        self, station_name: str
    ) -> list[float]:
        """Read atmospheric conditions at a specific station

        Parses the V3GR file to extract the atmospheric conditions for a station, and returns them as a list with the following items:
        1. Modified Julian Date
        2. Atmospheric pressure (hPa)
        3. Temperature (Celsius)
        4. Water vapor pressure (hPa)

        :param station_name: Name of the station
        :return site_data: Atmospheric conditions for the station
        """

        # Find the line in the V3GR file that contains the station data
        station_line = self.__find_station_line_in_v3gr_file(station_name)

        # Extract MJD and atmospheric conditions from line
        line_contents = station_line.split()[1:]
        atmospheric_conditions = [float(line_contents[0])]
        for item in line_contents[5:8]:
            atmospheric_conditions.append(float(item))

        return atmospheric_conditions
