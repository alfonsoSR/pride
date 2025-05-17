from pride import io, types
from pathlib import Path
import os
import pytest
from astropy import time
from datetime import datetime
import numpy as np
from pride.io.vex.interface import ScanData, SourceData


class TestVexInterface:

    vex_file = Path(__file__).parent.parent / "data/GR035.vix"

    def test_metadata(self) -> None:
        """Test Vex constructor"""

        vex = io.Vex(self.vex_file)

        # Check that the file was loaded correctly
        assert vex.experiment_name == "GR035"
        assert (vex.experiment_start - time.Time("2013-12-28T17:40:00.000")).to(
            "s"
        ).value == 0  # type: ignore
        assert (vex.experiment_stop - time.Time("2013-12-29T18:30:00.000")).to(
            "s"
        ).value == 0  # type: ignore

        return None

    def test_observation_modes(self) -> None:
        # TODO: Just compares against the original implementation, but it
        # doesn't actually check if it is correct

        vex = io.Vex(self.vex_file)
        _content = vex._Vex__content  # type: ignore

        new_modes = vex.load_observation_modes()

        old_modes = {
            mode: types.ObservationMode(
                mode, _content["MODE"][mode].getall("FREQ"), _content["FREQ"]
            )
            for mode in _content["MODE"]
        }

        # Check that the IDs are the same
        assert list(new_modes.keys()) == list(old_modes.keys())
        assert all(new_modes[key] == old_modes[key] for key in new_modes)

        return None

    def test_clock_parameters_interface(self) -> None:

        vex = io.Vex(self.vex_file)
        clock_parameters = vex.load_clock_parameters()

        assert clock_parameters is not None
        assert list(clock_parameters.keys())[0] == "Bd"
        assert (
            clock_parameters["Bd"][0] - datetime(2013, 12, 29, 6, 16)
        ).total_seconds() == 0
        assert np.isclose(clock_parameters["Bd"][1], 214.2550e-6)
        assert len(clock_parameters.keys()) == 29

        return None

    @pytest.mark.parametrize(
        ["ignored_stations", "expected_stations"],
        [
            (None, 29),
            (["Cd"], 28),
        ],
        ids=[
            "No ignored stations",
            "Cd ignored",
        ],
    )
    def test_load_stations_section(
        self, ignored_stations: list[str] | None, expected_stations: int
    ) -> None:

        vex = io.Vex(self.vex_file)
        stations_dictionary = vex.load_station_ids_and_names(ignored_stations)

        # Check for expected number of stations loaded
        assert len(stations_dictionary) == expected_stations

        # Check that the name of Ww is normalized
        assert stations_dictionary["Ww"] == "WARK12M"

        # Check that ignored stations are not present
        if ignored_stations is not None:
            for station in ignored_stations:
                assert station not in stations_dictionary

        # Check that incorrect VEX with multiple IDs for same station fails
        # The stations also have different names, but they should be normalized
        # and then the ID will be duplicated
        wrong_vex = io.Vex(self.vex_file.parent / "mock.vix")
        with pytest.raises(SystemExit) as error:
            wrong_vex.load_station_ids_and_names()

        return None

    @pytest.mark.parametrize(
        [
            "scan_id",
            "source_name",
            "reference_epoch",
            "expected_stations",
            "expected_offsets_Cd",
        ],
        [
            (
                "No0016",
                "J1232-0224",
                datetime(2013, 12, 28, 18, 24, 30),
                ["Cd", "Hb", "Yg", "Ke", "Ww", "Ym", "T6", "Km", "Ku"],
                (0, 120),
            ),
            (
                "No0017",
                "mex",
                datetime(2013, 12, 28, 18, 27, 00),
                ["Cd", "Hb", "Yg", "Ke", "Ww", "Ym", "T6", "Km", "Ku"],
                (0, 120),
            ),
        ],
        ids=[
            "Calibrator",
            "Target",
        ],
    )
    def test_load_scan_from_id(
        self,
        scan_id: str,
        source_name: str,
        reference_epoch: datetime,
        expected_stations: list[str],
        expected_offsets_Cd: tuple[int, int],
    ) -> None:

        # Load all scan IDs from VEX file
        vex = io.Vex(self.vex_file)

        # Load scan data from VEX file
        scan_data = vex.load_single_scan_data(scan_id, "mex")
        assert isinstance(scan_data, ScanData)

        # Check that the scan metadata is correct
        assert scan_data.source_name == source_name
        assert scan_data.initial_epoch == reference_epoch
        assert list(scan_data.offsets_per_station.keys()) == expected_stations

        # Check that the offsets for Ceduna are correct
        assert scan_data.offsets_per_station["Cd"] == expected_offsets_Cd

        return None

    @pytest.mark.parametrize(
        [
            "source_name",
            "expected_type",
            "fails",
            "use_mock_vex",
        ],
        [
            ("J1222+0413", "calibrator", False, False),  # Calibrator
            ("M362-2020", "target", False, False),  # Target
            ("mex", "", True, False),  # Fails: Name not found
            ("J1222+0413", "", True, True),  # Fails: Invalid type
            ("J1230+1223", "", True, True),  # Fails: Type information missing
            ("J1232-0224", "calibrator", True, True),  # Fails: Invalid frame
        ],
        ids=[
            "Calibrator",
            "Target",
            "Fails: Name not found",
            "Fails: Invalid type",
            "Fails: Type information missing",
            "Fails: Invalid frame",
        ],
    )
    def test_load_source_data(
        self,
        source_name: str,
        expected_type: str,
        fails: bool,
        use_mock_vex: bool,
    ) -> None:

        if use_mock_vex:
            vex = io.Vex(self.vex_file.parent / "mock.vix")
        else:
            vex = io.Vex(self.vex_file)

        if fails:
            with pytest.raises(SystemExit):
                _ = vex.load_source_data(source_name)
            return None

        source_data = vex.load_source_data(source_name)

        assert isinstance(source_data, SourceData)
        assert source_data.name == source_name
        assert source_data.source_type == expected_type

        raise NotImplementedError(
            "Missing verification of conversion of RA and DEC"
        )

        return None
