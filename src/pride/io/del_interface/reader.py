import struct
from pathlib import Path
import numpy as np
from ...logger import log
import sys
from dataclasses import dataclass
from .data_structures import DelContents, Scan
from typing import Any
from ... import utils


class DelFileInterface:
    """Interface to read binary DEL files

    The SFXC correlator reads estimated delays from binary files with .del extension. This class provides an interface to load information from DEL files.

    :param file: Path to the DEL file
    :param max_size_id: Maximum allowed size of an ID (scan, source...)
    """

    max_size_id: int = 80
    """Maximum allowed size of an ID: scan, source..."""

    def __init__(self, file: str | Path) -> None:
        """Constructor for DelFileInterface class

        :param file: Path to the DEL file
        """

        # Turn input into path
        self.file = Path(file).resolve()

        # Ensure that the file exists
        if not self.file.exists():
            log.error(
                "Failed to initialize DEL file interface: "
                f"{self.file.name} does not exist"
            )
            exit(1)

        # Ensure that the file has a .del extension
        self.file = Path(file).resolve()
        if self.file.suffix != ".del":
            log.error(
                "Failed to initialize DEL file interface: "
                f"{self.file} does not have a .del extension"
            )
            exit(1)

        # Ensure machine is little-endian
        if sys.byteorder != "little":
            log.error(
                "DEL file interface is not available on big-endian machines."
            )
            exit(1)

        # Load contents into a buffer of bytes
        self.buffer: bytes = self.file.read_bytes()
        self.buffer_size: int = len(self.buffer)

        return None

    def get_header_size(self, buffer: bytes) -> tuple[int, int]:
        """Get size of the header

        Reads the first 4 bytes of the DEL file to get the size of the header.

        :param buffer: Contents of the DEL file as a buffer of bytes
        :return header_size: Size of the header in bytes
        :return current_byte: Position from which to start reading the buffer
        """

        # Get size of header
        header_size = int.from_bytes(buffer[0:4], "little")

        # Set starting position in buffer to the size of the integer
        current_byte: int = 4

        return header_size, current_byte

    def read_contents(self) -> "DelContents":
        """Load contents of DEL file into a data structure

        :return: DelContents object with the contents of the DEL file
        """

        # Get header size, and initialize indicator of position in buffer
        (header_size,), current_byte = utils.peek_buffer(self.buffer, "<i", 0)
        assert isinstance(header_size, int)

        # Read station ID
        (station_id,), current_byte = utils.peek_buffer(
            self.buffer, f"<{header_size}s", current_byte
        )
        assert isinstance(station_id, str)

        # Loop over scan section and read data
        scans_list: list[Scan] = []
        while current_byte <= self.buffer_size:

            # Read scan ID
            (scan_id,), current_byte = utils.peek_buffer(
                self.buffer, f"<{self.max_size_id}sx", current_byte
            )
            assert isinstance(scan_id, str)

            # Read source name
            (source_name,), current_byte = utils.peek_buffer(
                self.buffer, f"<{self.max_size_id}sx", current_byte
            )
            assert isinstance(source_name, str)

            # Read date component of MJD
            (mjd_date,), current_byte = utils.peek_buffer(
                self.buffer, "<i", current_byte
            )
            assert isinstance(mjd_date, int)

            # Read data for current scan
            scan_data: list[list[float]] = []
            while True:

                # Read line with data
                values, current_byte = utils.peek_buffer(
                    self.buffer, "<7d", current_byte
                )

                # Check if end of scan section
                if len(values) == 0:
                    break

                # Append values to scan data
                scan_data.append(list(values))

            # Pack in data structure and append to list
            scans_list.append(
                Scan(scan_id, source_name, mjd_date, *np.array(scan_data).T)
            )

        return None
