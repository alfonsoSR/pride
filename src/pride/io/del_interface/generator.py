from pathlib import Path
import struct
import numpy as np
from ...logger import log
import sys


class DelFileGenerator:
    """Generate a DEL file with estimated delays

    The SFXC correlator reads estimated delays from binary files with .del extension. This class provides an interface to create such files with the outputs of a delay estimation.

    :param file: Path to the DEL file to be created
    :param max_size_id: Maximum allowed size of an ID (scan, source...)
    :param station_id_size: Size of the station ID in bytes
    :param header_size: Size of the header in bytes
    """

    max_size_id: int = 80
    """Maximum allowed size of an ID: scan, source..."""
    station_id_size: int = 2
    """Size of the station ID in bytes"""
    header_size: int = 11
    """Size of the header in bytes"""

    def __init__(self, del_file: str | Path) -> None:
        """Constructor for DelFileGenerator class

        :param del_file: Path to the DEL file to be created
        """

        # Ensure that the file has a .del extension
        self.file = Path(del_file).resolve()
        if self.file.suffix != ".del":
            log.error(
                "Failed to initialize interface for DEL file: "
                f"{self.file.name} does not have a .del extension"
            )
            exit(1)

        # If file exists, raise warning indicating that it will be overwritten
        if self.file.exists():
            log.warning(f"Overwriting existing DEL file: {self.file}")
            self.file.unlink()

        # Ensure machine is little-endian
        if sys.byteorder != "little":
            log.error(
                "DEL file generation is not supported on big-endian machines."
            )
            exit(1)

        # Initialize flags of internal state
        self.__has_header: bool = False
        self.__current_scan_is_validated: bool = False

        return None

    def __enter__(self) -> "DelFileGenerator":
        """Context manager interface for DelFileGenerator class"""

        log.info(f"Generating DEL file: {self.file}")
        self.file.touch()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit method for DelFileGenerator class"""

        if exc_type is not None:
            log.error(f"Failed to generate {self.file}")
            self.file.unlink()
        else:
            log.info(f"Succesfully generated {self.file}")

        return None

    def add_header(self, station_id: str, trailing_padding: int = 1) -> None:
        """Add header to DEL file

        The header is a set of 4 + header_size bytes. The first 4 bytes are an integer with the size of the remaining part of the header (header_size). The remaining bytes are occupied by a two-character station ID, and padding.

        After checking that the file is empty, the function initializes it by adding the header.

        :param station_id: Station ID
        :param trailing_padding: Size of the padding after the station ID
        """

        # Fail if the file is not empty
        if self.__has_header:
            log.error(
                f"Failed to generate {self.file.name}: "
                "Attempted to add header to non-empty file"
            )

        # Ensure that the station ID is valid
        if not len(station_id) == 2:
            log.error(
                f"Failed to generate {self.file.name}: "
                f"Invalid station ID {station_id}, should be "
                f"{self.station_id_size} characters"
            )
            exit(1)

        # Create header contents
        header_padding = (
            self.header_size - self.station_id_size - trailing_padding
        )
        header_format = f"<i{header_padding}x{self.station_id_size}s"
        if header_padding > 0:
            header_format += f"{header_padding}x"
        header_content = struct.pack(
            f"<i{header_padding}x{self.station_id_size}s{trailing_padding}x",
            self.header_size,
            station_id.encode("utf-8"),
        )

        # Write the header to the file
        with self.file.open("wb") as f:
            f.write(header_content)

        # Update internal state to indicate that the header has been added
        self.__has_header = True

        return None

    def validate_scan_contents(
        self, scan_id: str, source: str, mjd_ref: int, data: np.ndarray
    ) -> None:
        """Validate scan contents before writing to the DEL file

        Performs the following checks:
        - `scan_id` and `source` are strings with a maximum of 80 characters
        - `mjd_ref` is an integer
        - `data` is a 2D numpy array with shape (N, 7)

        The function terminates with an error if any of the checks fail.

        :param scan_id: Scan ID
        :param source: Source name
        :param mjd_ref: Modified Julian Date reference
        :param data: Data array with shape (N, 7)
        """

        # Initialize common error message
        common_error: str = f"While writing to {self.file.name}: "

        # Check type and length of scan_id
        assert isinstance(scan_id, str)
        if len(scan_id) > self.max_size_id:
            log.error(
                common_error
                + f"Scan ID {scan_id} is exceeds {self.max_size_id} characters"
            )
            exit(1)

        # Check type and length of source
        assert isinstance(source, str)
        if len(source) > self.max_size_id:
            log.error(
                common_error
                + f"Source name {source} exceeds {self.max_size_id} characters"
            )
            exit(1)

        # Check type of mjd_ref
        if not isinstance(mjd_ref, (int, np.integer)):
            log.error(
                common_error + f"MJD reference {mjd_ref} is not an integer"
            )
            exit(1)

        # Check type and shape of data
        assert isinstance(data, np.ndarray)
        if data.ndim != 2 or data.shape[1] != 7:
            log.error(
                common_error + f"Data array shape {data.shape} is not (N, 7)"
            )
            exit(1)

        # Set internal indicator to the last scan checked
        self.__last_scan_checked = scan_id

        return None

    def pack_id(self, id: str) -> bytes:
        """Generate bytes object for ID

        Generates a bytes object for an ID with a maximum size of 80 characters by filling the unused space with padding bytes. A padding byte is always added after the ID.

        :param id: ID to be packed
        :return: Packed bytes object
        """

        return struct.pack(f"<{self.max_size_id}sx", id.encode("utf-8"))

    def pack_scan_contents(
        self,
        scan_id: str,
        source_name: str,
        mjd_reference: int,
        scan_data: np.ndarray,
    ) -> bytes:
        """Pack scan contents into an array of bytes

        The function turns the scan contents into an array of bytes with the correct format and padding. An internal state flag is used to ensure that the scan data has been validated before calling this function.

        :param scan_id: Scan ID
        :param source: Source name
        :param mjd_ref: Modified Julian Date reference
        :param data: Data array with shape (N, 7)
        :return: Packed bytes object
        """

        # Ensure contents have been validated
        if not self.__current_scan_is_validated:
            log.error(
                f"Failed to generate {self.file}: "
                "Attempted to pack non-validated scan data"
            )
            exit(1)

        # Pack scan ID, source name, and MJD reference
        packed_data = self.pack_id(scan_id)
        packed_data += self.pack_id(source_name)
        packed_data += struct.pack("<i", mjd_reference)

        # Update packed data with scan information
        for data_item in scan_data:
            packed_data += struct.pack("<7d", *data_item)

        # Add line of zeros to indicate end of scan
        packed_data += struct.pack("<7d", *([0.0] * 7))

        return packed_data

    def add_scan(
        self,
        scan_id: str,
        source_name: str,
        mjd_reference: int,
        scan_data: np.ndarray,
    ) -> None:
        """Add scan data to pre-initialized DEL file

        After checking that the file has a header, and that the scan data has the correct format, the function packs the information into byte objects and appends it to the DEL file. An internal state flag is used to ensure that the scan data is validated before packing.

        :param scan_id: Scan ID [Maximum size: 80 characters]
        :param source_name: Source name [Maximum size: 80 characters]
        :param mjd_reference: MJD reference
        :param scan_data: Data to be added to the DEL file
        """

        # Fail if the file doesn't have a header
        if not self.__has_header:
            log.error(
                f"Failed to generate {self.file}: "
                "Attempted to add scan to DEL file without header"
            )
            exit(1)

        # Validate input
        self.validate_scan_contents(
            scan_id, source_name, mjd_reference, scan_data
        )
        self.__current_scan_is_validated = True

        # Pack scan contents
        packed_contents = self.pack_scan_contents(
            scan_id, source_name, mjd_reference, scan_data
        )

        # Write packed data to the file
        with self.file.open("ab") as f:
            f.write(packed_contents)

        # Require validation for the next scan
        self.__current_scan_is_validated = False

        return None
