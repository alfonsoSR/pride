from pathlib import Path
from ...logger import log
import numpy as np
from typing import Iterator
from scipy import interpolate


class IonexInterface:
    """Interface to content of IONEX files"""

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

    def read_data_from_ionex_file(
        self,
    ) -> tuple[list[interpolate.RegularGridInterpolator], float, float]:
        """Read data from IONEX file

        Parses the IONEX file to extract the reference height of the ionospheric model, the reference radius for the Eearth, and the TEC maps. The function generates a regular grid interpolator for each TEC map, and returns them all into a list, sorted by order of appearence in the file.

        :return tec_interpolators: List of interpolators for TEC maps
        :return ref_height: Reference height of the ionospheric model
        :return ref_rearth: Reference radius of the Earth
        """

        # Initialize output containers
        latitude_grid: np.ndarray | None = None
        longitude_grid: np.ndarray | None = None
        ref_height: float = NotImplemented
        ref_rearth: float = NotImplemented
        tec_maps_list: list[np.ndarray] = []

        # Iterate over the IONEX file to read all its TEC maps
        with self.ionex_file.open() as f:
            content = iter([line.strip() for line in f])

            # Parse content
            while True:

                # Read next line
                try:
                    line = next(content)
                except StopIteration:
                    break

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
