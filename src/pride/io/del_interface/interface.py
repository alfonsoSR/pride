from pathlib import Path
import struct
import numpy as np
from ...logger import log
from dataclasses import dataclass
from astropy import time
from ... import utils


@dataclass
class Scan:

    id: str
    source: str
    mjd_ref: int
    mjd2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    delays: np.ndarray
    doppler_phase: np.ndarray
    doppler_amp: np.ndarray


class DelFile:
    """Interface for DEL files: Binary file format for SFXC"""

    max_size_id: int = 80
    """Maximum allowed size of an ID: scan, source..."""

    format = {
        "header": b"<i2sx",
        "source": b"<80sxi",
        "data": b"<7d",
    }
    length = {
        "header": 7,
        "source": 85,
        "data": 56,
    }

    def __init__(self, file: str | Path) -> None:

        # Ensure that the file has a .del extension
        self.file = Path(file).resolve()
        if self.file.suffix != ".del":
            log.error(
                "Failed to initialize interface for DEL file: "
                f"{self.file.name} does not have a .del extension"
            )
            exit(1)

        return None

    @property
    def exists(self) -> bool:
        return self.file.is_file()

    def create_file(self, station_id: str) -> None:
        """Create a new DEL file with just the header"""

        # Ensure that the
        if self.exists:
            log.warning(f"Overwriting existing DEL file: {self.file}")

        with self.file.open("wb") as f:

            # Generate header using 11 bytes of size
            header = struct.pack("<i8x2sx", 11, station_id.encode("utf-8"))

            # Add header to file
            f.write(header)

        return None

    def validate_data_format(
        self, scan_id: str, source: str, mjd_ref: int, data: np.ndarray
    ) -> None:
        """Validate the data format before writing to the DEL file

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

        return None

    def __pack_string_with_padding(self, string: str, max_size: int) -> bytes:

        string_size = len(string)
        string_fmt = bytes(f"<{max_size}sx", "utf-8")
        string_bytes = struct.pack(string_fmt, string.encode("utf-8"))

        return string_bytes

    def add_scan(
        self, scan_id: str, source: str, mjd1: int, data: np.ndarray
    ) -> None:
        """Add a scan information to DEL file"""

        # Ensure that the DEL file exists
        if not self.exists:
            log.error(
                "Attempted to add scan to non-existing DEL file"
                f" {self.file.name}"
            )
            exit(1)

        # Validate data format
        self.validate_data_format(scan_id, source, mjd1, data)

        # Pack scan ID and source name
        scan_id_bytes = self.__pack_string_with_padding(
            scan_id, self.max_size_id
        )
        source_bytes = self.__pack_string_with_padding(source, self.max_size_id)

        # Write data to file
        with self.file.open("ab") as f:

            # Write scan ID
            f.write(scan_id_bytes)

            # Write source ID
            f.write(source_bytes)

            # Write MJD reference
            mjd_ref_bytes = struct.pack(b"<i", mjd1)
            f.write(mjd_ref_bytes)

            # Write data
            for values in data:

                # Pack value
                packed_values = struct.pack(b"<7d", *values)
                f.write(packed_values)

            # Write empty line to indicate end of scan
            packed_end_of_scan = struct.pack(b"<7d", *([0.0] * 7))
            f.write(packed_end_of_scan)

        return None

    def read_station_id(self, data: bytes) -> tuple[str, int]:
        """Get station id from the header of the del file

        The DEL file starts with an integer, which defines the size of the header without taking the integer itself (4 bytes) into account. The structure of a header with size X bytes is:

        - (X - 3) pad bytes
        - 2 byte string with station code
        - 1 pad byte after the string

        This function reads the header size, skips the padding bytes, reads and decodes the station ID, and returns it as a string, leaving the `byte` indicator right at the start of the next section of the file.

        :param data: Buffer with del file contents
        :return station_id: Station id as two letter code
        :return byte: Byte from which to keep reading the file
        """

        # Get length of header
        (header_length,), byte = self.peek(data, b"<i", 0)
        assert isinstance(header_length, int)

        # Skip padding bytes
        byte += header_length - 3

        # Read station ID
        (station_id,), byte = self.peek(data, b"<2sx", byte)
        assert isinstance(station_id, str)

        return station_id, byte

    def read(self) -> tuple[str, list[Scan]]:
        """Read data from DEL file"""

        # Ensure that file exists
        if not self.exists:
            log.error(
                f"Failed to read DEL file: {self.file.name} does not exist"
            )
            exit(1)

        # Load binary data from file
        with self.file.open("rb") as f:
            data = f.read()
            size = len(data)

        # Read station ID and move to start of data section
        station_id, byte = self.read_station_id(data)

        # Loop over scans
        scans: list["Scan"] = []
        while byte < size:

            # Read scan ID and skip padding
            (scan_id,), byte = self.peek(data, b"<80sx", byte)
            assert isinstance(scan_id, str)

            # Read source ID and skip padding
            (source_id,), byte = self.peek(data, b"<80sx", byte)
            assert isinstance(source_id, str)

            # Read MJD for scan
            (mjd_ref,), byte = self.peek(data, b"<i", byte)
            assert isinstance(mjd_ref, int)

            # Read data for current scan
            scan_data: list[list[float]] = []
            while True:

                # Read line with data
                values, byte = self.peek(data, b"<7d", byte)

                # If line is empty, this is the end of the scan
                if sum(values) == 0:
                    break

                # Append values to scan data
                scan_data.append(values)

            # Initialize data structure for current scan
            current_scan = Scan(
                scan_id, source_id, mjd_ref, *np.array(scan_data).T
            )

            # Add scan to list
            scans.append(current_scan)

        return station_id, scans

    def peek(self, data: bytes, format: bytes, start: int) -> tuple[list, int]:
        """Read and decode array of bytes from file"""

        # Get length of requested format
        length: int = struct.calcsize(format)

        # Unpack content
        content = struct.unpack(format, data[start : start + length])

        # Decode output
        output = []
        for item in content:
            if isinstance(item, bytes):

                # Decode bytes to string
                item_str: str = item.decode("utf-8")

                # Remove white-space padding
                item_str = item_str.replace(" ", "")

                # Remove null characters
                item_str = item_str.replace("\x00", "")

                # Add to output
                output.append(item_str)
            else:
                output.append(item)

        return output, start + length
