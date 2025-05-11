from pathlib import Path
import struct
import numpy as np
from ...logger import log
from dataclasses import dataclass


@dataclass
class Scan:

    id: int
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

        self.file = Path(file).resolve()

        return None

    @property
    def exists(self) -> bool:
        return self.file.is_file()

    def create_file(self, station_id: str) -> None:
        """Create a new DEL file with just the header"""

        if self.exists:
            log.warning(f"Overwriting existing DEL file: {self.file.name}")

        with self.file.open("wb") as f:
            f.write(struct.pack(self.format["header"], 3, station_id.encode()))

        return None

    def add_scan(self, source: str, mjd1: int, data: np.ndarray) -> None:
        """Add a scan information to DEL file"""

        # Ensure data has the right shape
        if (data.ndim != 2) or (data.shape[1] != 7):
            log.error(
                f"While writing to {self.file.name}: "
                f"Invalid shape for data array :: {data.shape} != (N, 7)"
            )
            exit(1)

        # Ensure that file exists
        if not self.exists:
            log.error(f"File {self.file.name} does not exist")
            exit(1)

        # Write data to file
        with self.file.open("ab") as f:
            f.write(struct.pack(self.format["source"], source.encode(), mjd1))
            for values in data:
                f.write(struct.pack(self.format["data"], *values))
            f.write(struct.pack(self.format["data"], *([0.0] * 7)))

        return None

    def read(self) -> tuple[list, list[Scan]]:
        """Read data from DEL file"""

        # Ensure that file exists
        if not self.exists:
            log.error(f"File {self.file.name} does not exist")
            exit(1)

        # Load binary data from file
        with self.file.open("rb") as f:
            data = f.read()
            size = len(data)

        # Read header
        header, byte = self.peek(
            data, self.format["header"], self.length["header"], 0
        )

        # Loop over scans
        scans: list["Scan"] = []
        scan_number = 0
        while byte < size:

            # Read source and ref mjd for scan
            (source, mjd_ref), byte = self.peek(
                data, self.format["source"], self.length["source"], byte
            )

            # Read data for scan
            scan_data = []
            while True:
                values, byte = self.peek(
                    data, self.format["data"], self.length["data"], byte
                )
                if sum(values) == 0:
                    break
                scan_data.append(values)

            # Add scan to list
            current_scan = Scan(
                scan_number, source, mjd_ref, *np.array(scan_data).T
            )
            scans.append(current_scan)
            scan_number += 1

        return header, scans

    def peek(
        self, data: bytes, format: bytes, length: int, start: int
    ) -> tuple[list, int]:
        """Read and decode array of bytes from file"""

        content = struct.unpack(format, data[start : start + length])
        output = []
        for item in content:
            if isinstance(item, bytes):
                output.append(item.decode("utf-8").rstrip("\x00"))
            else:
                output.append(item)

        return output, start + length
