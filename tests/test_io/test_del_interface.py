from pride.io.del_interface.generator import DelFileGenerator
from pride.io.del_interface.reader import DelFileInterface
from pathlib import Path
import pytest
import os
import numpy as np
import struct


class TestDelFileInterface:
    """Tests for functionality to read data from DEL files."""

    del_file: Path = Path(__file__).parent.parent / "data/del_from_sfxc.del"

    def test_initialize_del_object(self) -> None:

        # Initialize interface for valid DEL file
        interface = DelFileInterface(self.del_file)

        # Check that fails when file does not exist
        with pytest.raises(SystemExit):
            _ = DelFileInterface("non_existent.del")

        # Check that fails when file exists, but does not have .del extension
        with pytest.raises(SystemExit):
            _ = DelFileInterface("GR035.vix")

        return None

    def test_peek(self) -> None:

        return None


class TestDelFileGenerator:
    """Tests for functionality to create DEL files."""

    mock_del: str = "mock.del"

    def test_initialize_del_generator(self, tmp_path: Path) -> None:

        # Initialize generator for valid DEL file
        generator = DelFileGenerator(self.mock_del)

        # Try to initialize generator for file with incorrect extension
        with pytest.raises(SystemExit):
            _ = DelFileGenerator("GR035.vix")

        # Check that existing DEL file would be overwritten
        mock_del_path = tmp_path / self.mock_del
        mock_del_path.touch()
        assert mock_del_path.exists()
        generator = DelFileGenerator(mock_del_path)
        assert not generator.file.exists()

        return None

    def test_add_header(self, tmp_path: Path) -> None:

        mock_del_path = tmp_path / self.mock_del

        # Generate DEL file with header
        with DelFileGenerator(mock_del_path) as generator:
            generator.add_header("Ef")
        assert mock_del_path.exists()

        # Manually check content of the header
        with mock_del_path.open("rb") as f:

            contents = f.read()
            assert len(contents) == 4 + DelFileGenerator.header_size
            assert contents[4:] == b"\x00\x00\x00\x00\x00\x00\x00\x00Ef\x00"

        return None

    @pytest.mark.parametrize(
        ["scan_id", "source_name", "mjd_ref", "scan_data", "fails"],
        [
            (
                "No0001",
                "CALIBRATOR",
                60218,
                np.random.rand(2, 7),
                False,
            ),
            (
                "N" * DelFileGenerator.max_size_id,
                "C" * DelFileGenerator.max_size_id,
                np.int32(60218),
                np.random.rand(2, 7),
                False,
            ),
            (
                "N" * (DelFileGenerator.max_size_id + 1),
                "CALIBRATOR",
                60218,
                np.random.rand(2, 7),
                True,
            ),
            (
                "No0001",
                "C" * (DelFileGenerator.max_size_id + 1),
                60218,
                np.random.rand(2, 7),
                True,
            ),
            (
                "No0001",
                "CALIBRATOR",
                60218,
                np.random.rand(7),
                True,
            ),
            (
                "No0001",
                "CALIBRATOR",
                60218,
                np.random.rand(2, 6),
                True,
            ),
        ],
        ids=[
            "Valid",
            "Valid: Valid corner cases",
            "Invalid: Scan ID too long",
            "Invalid: Source ID too long",
            "Invalid: Data array not 2D",
            "Invalid: Incorrect amount of columns",
        ],
    )
    def test_validate_scan_contents(
        self,
        scan_id: str,
        source_name: str,
        mjd_ref: int,
        scan_data: np.ndarray,
        fails: bool,
        tmp_path: Path,
    ) -> None:
        """Test utility to ensure that contents of a scan are valid"""

        interface = DelFileGenerator(tmp_path / self.mock_del)
        if fails:
            with pytest.raises(SystemExit):
                interface.validate_scan_contents(
                    scan_id, source_name, mjd_ref, scan_data
                )
        else:
            interface.validate_scan_contents(
                scan_id, source_name, mjd_ref, scan_data
            )

        return None

    def test_pack_scan_id(self, tmp_path: Path) -> None:

        interface = DelFileGenerator(tmp_path / self.mock_del)
        packed_scan_id = interface.pack_id("No0001")
        assert packed_scan_id == b"No0001" + b"\x00" * 75

    def test_pack_scan_content(self, tmp_path: Path) -> None:

        interface = DelFileGenerator(tmp_path / self.mock_del)

        # Generate synthetic data
        scan_contents = (
            "No0001",
            "CALIBRATOR",
            60218,
            np.array([[1, 2, 3, 4, 5, 6, 7]]),
        )

        # Ensure packing fails if validation is not done
        with pytest.raises(SystemExit):
            _ = interface.pack_scan_contents(*scan_contents)

        # Simulate validation by manually setting state flag
        interface._DelFileGenerator__current_scan_is_validated = True  # type: ignore

        packed_contents = interface.pack_scan_contents(*scan_contents)
        assert len(packed_contents) == 2 * 81 + 4 + 14 * 8
        assert packed_contents[0:81] == b"No0001" + b"\x00" * 75
        assert packed_contents[81:162] == b"CALIBRATOR" + b"\x00" * 71
        assert packed_contents[162:166] == int(60218).to_bytes(4, "little")
        assert packed_contents[166:222] == struct.pack(
            "<7d", *scan_contents[3][0]
        )
        assert packed_contents[222:] == b"\x00" * 7 * 8

        return None

    def test_add_scan(self, tmp_path: Path) -> None:

        # Generate synthetic data
        station_id = "Ld"
        scan_contents = (
            "No0001",
            "CALIBRATOR",
            60218,
            np.array([[1, 2, 3, 4, 5, 6, 7]]),
        )

        # Create DEL file with header and scan
        del_path = tmp_path / self.mock_del
        with DelFileGenerator(del_path) as generator:
            generator.add_header(station_id)
            generator.add_scan(*scan_contents)

        # Check that the DEL file exists
        assert del_path.exists()
        with del_path.open("rb") as f:

            contents = f.read()
            assert (
                len(contents)
                == 4 + DelFileGenerator.header_size + 81 + 81 + 4 + 14 * 8
            )
            assert contents[0:4] == DelFileGenerator.header_size.to_bytes(
                4, "little"
            )
            assert contents[4 : 4 + DelFileGenerator.header_size] == (
                b"\x00" * (DelFileGenerator.header_size - 3) + b"Ld\x00"
            )
            offset = DelFileGenerator.header_size + 4
            assert contents[offset : offset + 81] == b"No0001" + b"\x00" * 75
            assert (
                contents[81 + offset : offset + 162]
                == b"CALIBRATOR" + b"\x00" * 71
            )
            assert contents[offset + 162 : offset + 166] == int(60218).to_bytes(
                4, "little"
            )
            assert contents[offset + 166 : offset + 222] == struct.pack(
                "<7d", *scan_contents[3][0]
            )
            assert contents[offset + 222 :] == b"\x00" * 7 * 8

            # Ensure that the internal state is reset after writing
            assert not generator._DelFileGenerator__current_scan_is_validated  # type: ignore
