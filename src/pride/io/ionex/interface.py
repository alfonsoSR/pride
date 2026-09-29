from pathlib import Path
from ...logger import log
import numpy as np
from typing import Iterator
from scipy import interpolate
from datetime import datetime
from astropy import time
from io import TextIOWrapper

class IonexInterface:
    """Interface to content of IONEX files"""

    supported_ionex_versions: list[str] = ["1.0", "1.1"]
    """List of supported versions of IONEX format"""

    def __init__(self, ionex_file: Path) -> None:
        """Constructor for IonexInterface class

        :param ionex_file: Path to the IONEX file
        """

        # Ensure the file exists
        self.ionex_file = ionex_file
        if not self.ionex_file.exists():
            log.error(
                f"Failed to parse IONEX file {self.ionex_file}: "
                "File does not exist"
            )
            exit(1)

        # Common part of error message
        self.base_error = f"Failed to parse IONEX file {self.ionex_file.name}: "

        # Load metadata from header section
        self.__has_metadata: bool = False
        self.__load_metadata_from_header()

        return None

    def __get_ref_height_from_line(self, ref_height: float, line: str) -> float:

        # Read reference ionospheric height
        if "HGT1 / HGT2 / DHGT" in line:

            # Read reference values for height
            h1, h2 = np.array(line.split()[:2], dtype=float)

            # Ensure that heights are equal
            if h1 != h2:
                log.error(
                    f"Unexpected content in IONEX file: "
                    f"Reference heights {h1} and {h2} do not match"
                )
                exit(1)

            # Update reference height
            ref_height = h1

        return ref_height

    def __get_ref_rearth_from_line(self, ref_rearth: float, line: str) -> float:

        # Read reference radius of the Earth
        if "BASE RADIUS" in line:
            ref_rearth = float(line.split()[0])

        return ref_rearth

    def __get_latitude_grid_from_line(
        self, latitude_grid: np.ndarray | None, line: str
    ) -> np.ndarray | None:

        # Read latitude grid
        if "LAT1 / LAT2 / DLAT" in line:
            l0, l1, dl = np.array(line.split()[:3], dtype=float)
            latitude_grid = np.arange(l0, l1 + dl / 2, dl)

        return latitude_grid

    def __get_longitude_grid_from_line(
        self, longitude_grid: np.ndarray | None, line: str
    ) -> np.ndarray | None:

        # Read longitude grid
        if "LON1 / LON2 / DLON" in line:
            l0, l1, dl = np.array(line.split()[:3], dtype=float)
            longitude_grid = np.arange(l0, l1 + dl / 2, dl)

        return longitude_grid

    def __read_tec_map(
        self,
        content: Iterator[str],
        line: str,
        latitude_grid: np.ndarray | None,
        longitude_grid: np.ndarray | None,
    ) -> tuple[Iterator[str], str, np.ndarray]:

        # Ensure that the latitude and longitude grids were found
        if latitude_grid is None or longitude_grid is None:
            log.error(
                f"Failed to read IONEX file: "
                "Attempted to read TEC map before defining grids"
            )
            exit(1)

        assert "START OF TEC MAP" in line  # Sanity

        # Skip two lines to reach the content section
        next(content)
        next(content)

        # Read the TEC map
        tec_map = np.zeros((len(latitude_grid), len(longitude_grid)))
        for i, _ in enumerate(latitude_grid):

            tec_map[i] = np.array(
                " ".join([next(content).strip() for _ in range(5)]).split(),
                dtype=float,
            )
            line = next(content)

        assert "END OF TEC MAP" in line

        return content, line, tec_map

    def __load_metadata_from_header(self) -> None:
        """Load IONEX metadata from header section into attributes

        Parses the header section of the file and loads all the non-optional metadata defined in appendix A of the IONEX Format Version 1.1 (Schaer, Gurtner, Feltens). This method is only meant to make the constructor more readable.
        """

        # Return with error if object has metadata
        if self.__has_metadata:
            log.error(
                self.base_error
                + f"Attempted to parse header when metadata is already present"
            )
            exit(1)

        with self.ionex_file.open() as buffer:

            # Initialize current line content
            line = buffer.readline()

            # Ensure ionex format version is supported
            ionex_version = line[:8].strip()
            if ionex_version not in self.supported_ionex_versions:
                log.error(
                    self.base_error
                    + f"IONEX version {ionex_version} is not supported"
                )
                exit(1)

            # Skip header description
            while line[60:80] != "EPOCH OF FIRST MAP  ":
                line = buffer.readline()

            # Read epoch of first TEC map
            self.initial_tec_epoch: str = datetime(
                year=int(line[:6]),
                month=int(line[6:12]),
                day=int(line[12:18]),
                hour=int(line[18:24]),
                minute=int(line[24:30]),
                second=int(line[30:36]),
            ).strftime(r"%Y-%m-%dT%H:%M:%S")

            # Read epoch of last TEC map
            line = buffer.readline()
            self.final_tec_epoch: str = datetime(
                year=int(line[:6]),
                month=int(line[6:12]),
                day=int(line[12:18]),
                hour=int(line[18:24]),
                minute=int(line[24:30]),
                second=int(line[30:36]),
            ).strftime(r"%Y-%m-%dT%H:%M:%S")

            # Read interval between ionex epochs
            self.tec_interval: int = int(buffer.readline()[:6])

            # Read number of TEC maps in file
            self.number_of_tec_maps: int = int(buffer.readline()[:6])

            # Read mapping function
            self.mapping_function: str = buffer.readline()[:6]

            # Read elevation cutoff
            self.elevation_cutoff: float = float(buffer.readline()[:8])

            # Skip 3 lines: Observables used, # stations, # satellites
            for _ in range(3):
                buffer.readline()

            # Reference Earth radius
            self.ref_rearth: float = float(buffer.readline()[:8])

            # Read map dimensions: Raise error if 3D (Not supported)
            tec_map_dimension: int = int(buffer.readline()[:6])
            if tec_map_dimension != 2:
                log.error(
                    self.base_error
                    + f"{tec_map_dimension}D TEC maps are not supported"
                )
                exit(1)

            # Read reference height of ionospheric model (Assumes 2D maps)
            self.ref_height: float = float(buffer.readline()[2:8])

            # Generate latitude grid
            lat_start, lat_end, dlat = np.array(
                buffer.readline()[2:20].split(), dtype=float
            )
            self.latitude_grid = np.arange(
                lat_start, lat_end + 0.5 * dlat, dlat
            )

            # Generate longitude grid
            lon_start, lon_end, dlon = np.array(
                buffer.readline()[2:20].split(), dtype=float
            )
            self.longitude_grid = np.arange(
                lon_start, lon_end + 0.5 * dlon, dlon
            )

            # Skip remaining header
            line = buffer.readline()
            while line[60:].strip() != "END OF HEADER":
                line = buffer.readline()

        # Set metadata flag to true
        self.__has_metadata = True

        return None

    def generate_tec_map_interpolators(
        self,
    ) -> dict[time.Time, interpolate.RegularGridInterpolator]:
        """Generate regular grid interpolators for TEC maps in file

        For each TEC map in the file, it loads the data into a numpy array, interpolates it using scipy.interpolate.RegularGridInterpolator, and then adds it to a dictionary with the UTC epoch of the TEC map as key.

        :return: Dictionary with UTC epochs of the TEC maps as keys, and regular grid interpolators for their contents as values.
        """

        # Initialize output container
        tec_map_dictionary: dict[
            time.Time, interpolate.RegularGridInterpolator
        ] = {}

        # Get sizes of latitude and longitude arrays
        latitude_grid_size: int = len(self.latitude_grid)
        longitude_grid_size: int = len(self.longitude_grid)

        # Number of lines to occupied by each row of the TEC map
        # Each line in the ionex file is 80 characters long, and each TEC value
        # is a 5 digit integer. Each row contains `longitude_grid_size` values,
        # so the number of lines to read is (longitude_grid_size * 5 // 80) + 1
        lines_per_tec_map: int = (longitude_grid_size * 5 // 80) + 1

        with self.ionex_file.open() as buffer:

            # Jump to the beginning of the first TEC map
            line = buffer.readline()
            while line[60:80].strip() != "END OF HEADER":
                line = buffer.readline()

            for map_index in range(self.number_of_tec_maps):

                # Skip line indicating start of map
                buffer.readline()

                # Update current epoch
                line = buffer.readline()
                current_epoch_str = datetime(
                    year=int(line[:6]),
                    month=int(line[6:12]),
                    day=int(line[12:18]),
                    hour=int(line[18:24]),
                    minute=int(line[24:30]),
                    second=int(line[30:36]),
                ).strftime(r"%Y-%m-%dT%H:%M:%S")
                current_epoch = time.Time(current_epoch_str, scale="utc")

                # Read current TEC map into array
                current_tec_map, buffer = self.__read_single_tec_map(
                    buffer,
                    latitude_grid_size,
                    longitude_grid_size,
                    lines_per_tec_map,
                )

                # Generate grid interpolator for current map
                current_interpolator = interpolate.RegularGridInterpolator(
                    (self.longitude_grid, self.latitude_grid), current_tec_map.T
                )

                # Update dictionary
                tec_map_dictionary[current_epoch] = current_interpolator

                # Skip line with end of map
                buffer.readline()

        return tec_map_dictionary

    def __read_single_tec_map(
        self,
        buffer: TextIOWrapper,
        latitude_grid_size: int,
        longitude_grid_size: int,
        lines_per_tec_map: int,
    ) -> tuple[np.ndarray, TextIOWrapper]:
        """Load content of a TEC map into a numpy array

        :param buffer: Text buffer pointing to the beginning of the data section of a TEC map (line after EPOCH OF CURRENT MAP)
        :param latitude_grid_size: Number of data points in the latitude discretization
        :param longitude_grid_size: Number of data points in the longitude discretization
        :param lines_per_tec_map: Number of lines occupied by the entry of the TEC map associated with a certain latitude
        :return tec_data: Array of TEC values groupped by latitude
        :return buffer: Buffer pointing to the line before END OF CURRENT TEC MAP
        """

        # Initialize array for TEC map
        current_tec_map: np.ndarray = np.zeros(
            (latitude_grid_size, longitude_grid_size)
        )
        for idx in range(latitude_grid_size):

            # Skip line with grid information
            buffer.readline()

            # Load data for current latitude into array
            current_tec_map[idx] = np.array(
                " ".join(
                    [
                        buffer.readline().strip()
                        for _ in range(lines_per_tec_map)
                    ]
                ).split(),
                dtype=int,
            )

        return current_tec_map, buffer

    def read_data_from_ionex_file(
        self,
    ) -> tuple[list[interpolate.RegularGridInterpolator], float, float]:
        """Read data from IONEX file

        Parses the IONEX file to extract the reference height of the ionospheric model, the reference radius for the Eearth, and the TEC maps. The function generates a regular grid interpolator for each TEC map, and returns them all into a list, sorted by order of appearence in the file.

        :return tec_interpolators: List of interpolators for TEC maps
        :return ref_height: Reference height of the ionospheric model
        :return ref_rearth: Reference radius of the Earth
        """

        raise DeprecationWarning("This method is deprecated")

        # Initialize output containers
        latitude_grid: np.ndarray | None = None
        longitude_grid: np.ndarray | None = None
        ref_height: float = NotImplemented
        ref_rearth: float = NotImplemented
        tec_maps_list: list[np.ndarray] = []

        # Iterate over the IONEX file to read all its TEC maps
        with self.ionex_file.open() as buffer:

            content = iter([line.rstrip() for line in f])

            # Parse content
            while True:

                # Read next line
                try:
                    line = next(content)
                except StopIteration:
                    break

                # Skip header description
                if line[60:] != "EPOCH OF FIRST MAP":
                    continue

                # Load initial and final epochs, and interval
                initial_epoch = datetime(
                    year=int(line[:6]),
                    month=int(line[6:12]),
                    day=int(line[12:18]),
                    hour=int(line[18:24]),
                    minute=int(line[24:30]),
                    second=int(line[30:36]),
                )
                line = next(content)

                # Read reference height of ionospheric model
                ref_height = self.__get_ref_height_from_line(ref_height, line)

                # Read reference radius of the Earth
                ref_rearth = self.__get_ref_rearth_from_line(ref_rearth, line)

                # Define latitude grid
                latitude_grid = self.__get_latitude_grid_from_line(
                    latitude_grid, line
                )

                # Define longitude grid
                longitude_grid = self.__get_longitude_grid_from_line(
                    longitude_grid, line
                )

                # Look for start of the data block
                if "START OF TEC MAP" not in line:
                    continue

                # Read TEC map
                content, line, tec_map = self.__read_tec_map(
                    content, line, latitude_grid, longitude_grid
                )

                # Add TEC map to output container
                tec_maps_list.append(tec_map)

        # Generate an interpolator for each TEC map
        tec_interpolators: list[interpolate.RegularGridInterpolator] = [
            interpolate.RegularGridInterpolator(
                (longitude_grid, latitude_grid), tec_map.T
            )
            for tec_map in tec_maps_list
        ]
        return tec_interpolators, ref_height, ref_rearth
