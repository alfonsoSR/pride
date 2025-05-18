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

    def read_v3gr_data_for_station(self, station_name: str) -> list[float]:
        """Read V3GR data for a specific station

        :param station_name: Name of the station
        :return site_data: List of coefficients for the station
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

        # Extract MJD and coefficients from line
        line_contents = station_line.split()
        coefficients = [float(x) for x in line_contents[1:6]] + [
            float(x) for x in line_contents[9:]
        ]
        return coefficients
