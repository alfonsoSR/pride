from pathlib import Path
import struct
import numpy as np
from datetime import datetime
from astropy import time
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

    def create_ascii_file(
        self,
        station_id: str,
        experiment_name: str = "unknown",
        baseline: str = "unknown",
        delay_model_info: str = "VMF1",
    ) -> None:
        """Create a new ASCII delay file with header"""

        ascii_file = self.file.with_suffix(".txt")

        if ascii_file.is_file():
            log.warning(f"Overwriting existing ASCII file: {ascii_file.name}")

        utc_now = datetime.utcnow().isoformat(sep=" ", timespec="microseconds") + " UTC"

        header = f"""# A priori delay table for station {station_id}, experiment {experiment_name}.
#
# Produced by pypride-asr software package on {utc_now}.
#
# Column descriptions:
#   1-6:   UTC time stamp.
#     7:   Source name.
#  8-10:   Far-field sources: [u v w],
#          computed numerically as c*[d(tau)/d(RA)/cos(D),
#                                     d(tau)/d(D),
#                                     tau],
#          c - speed of light in vacuum.
#          Near-field sources: [c*d(tau)/d(theta)/cos(phi),
#                               c*d(tau)/d(phi),
#                               d(tau)/d(rho)].
#    11:   Geometric delay.
#    12:   Delay due to thermal expansion of telescope(s).
#    13:   Delay due to axis offset.
#    14:   Tropospheric delay (model used: {delay_model_info}).
#    15:   Ionospheric delay (for base freq of first BBC).
# 16-17:   Phase [rad] and amplitude corrections applied.
#
# For details about software and models used therein, see
# Duev et al. A&A 541, A43 (2012), http://dx.doi.org/10.1051/0004-6361/201218885
#
"""

        with ascii_file.open("w") as f:
            f.write(header)

        return None

    def add_scan_ascii(
        self,
        source: str,
        mjd1: int,
        data: np.ndarray,
        delay_components: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Add scan information to ASCII delay file"""

        # Ensure data has the right shape
        if (data.ndim != 2) or (data.shape[1] != 7):
            log.error(
                f"While writing to ASCII file: "
                f"Invalid shape for data array :: {data.shape} != (N, 7)"
            )
            exit(1)

        ascii_file = self.file.with_suffix(".txt")

        # Ensure that file exists
        if not ascii_file.is_file():
            log.error(f"ASCII file {ascii_file.name} does not exist")
            exit(1)

        # Extract individual delay components if provided
        # The format expects: geometric, thermal, axis_offset, tropospheric, ionospheric
        geometric = np.zeros(len(data))
        thermal = np.zeros(len(data))
        axis_offset = np.zeros(len(data))
        tropospheric = np.zeros(len(data))
        ionospheric = np.zeros(len(data))

        if delay_components is not None:
            # Map delay model names to the expected components
            if "Geometric" in delay_components:
                geometric = delay_components["Geometric"]
            if "AntennaDelays" in delay_components:
                thermal = delay_components["AntennaDelays"]
            if "Tropospheric" in delay_components:
                tropospheric = delay_components["Tropospheric"]
            if "Ionospheric" in delay_components:
                ionospheric = delay_components["Ionospheric"]

        # Write data to ASCII file
        with ascii_file.open("a") as f:
            for i, values in enumerate(data):
                mjd2_sec, u, v, w, total_delay, doppler_phase, doppler_amp = values

                # Convert MJD (days + seconds) to UTC timestamp
                mjd_full = mjd1 + mjd2_sec / 86400.0
                t = time.Time(mjd_full, format="mjd")
                iso = t.iso.split()
                date_parts = iso[0].split("-")
                time_parts = iso[1].split(":")

                # Format: YYYY MM DD HH MM SS.sss
                timestamp = f"{date_parts[0]} {date_parts[1]} {date_parts[2]} {time_parts[0]} {time_parts[1]} {time_parts[2]}"

                # Write line with all components
                line = (
                    f"{timestamp:30s} {source:>15s}  "
                    f"{u:23.16e} {v:23.16e} {w:23.16e}  "
                    f"{total_delay:23.16e} "
                    f"{thermal[i]:13.7e} {axis_offset[i]:13.7e} "
                    f"{tropospheric[i]:13.7e} {ionospheric[i]:13.7e}  "
                    f"{doppler_phase:23.16e} {doppler_amp:23.16e}\n"
                )
                f.write(line)

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
