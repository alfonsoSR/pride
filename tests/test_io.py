from pride import io, types
from pathlib import Path
import os
import pytest
from astropy import time
from datetime import datetime
import numpy as np


class TestKernelManagement:
    """Tests for kernel management functionality in IO"""

    def test_constructor(self, tmp_path: Path) -> None:

        # Test constructor for ESA mission
        manager_esa = io.SpiceKernelManager("JUICE", tmp_path)
        assert manager_esa.target == "juice"
        assert manager_esa.kernels_folder.exists()
        assert "esa" in manager_esa._SpiceKernelManager__api  # type: ignore

        # Test constructor for NASA mission
        # TODO: We don't have any NASA missions in the catalog yet

        return None

    def test_metakernel_acquisition(self, tmp_path: Path) -> None:
        """Test metakernel acquisition"""

        manager = io.SpiceKernelManager("JUICE", tmp_path)

        # Metakernel not present: download
        metakernel = manager.ensure_metakernel()
        assert metakernel.exists()
        assert metakernel.name == "metak.tm"

        # Get Unix timestamp for the last modification time
        creation_time = os.path.getmtime(metakernel)

        # Check information in the metakernel
        with metakernel.open("r") as f:

            # Check that the path to the kernels has been updated
            for line in f:
                path_values_line_found: bool = False
                if line.strip().startswith("PATH_VALUES"):
                    path_values_line_found = True
                    assert str(manager.kernels_folder) in line
                    break
            assert path_values_line_found

        # Check that the metakernel is not downloaded again
        new_manager = io.SpiceKernelManager("JUICE", tmp_path)
        new_metakernel = new_manager.ensure_metakernel()
        assert new_metakernel.exists()
        assert new_metakernel.name == "metak.tm"
        assert os.path.getmtime(new_metakernel) == creation_time

        # Remove the metakernel for other tests
        metakernel.unlink()
        return None

    @pytest.mark.parametrize("mission", ["JUICE", "MARS-EXPRESS"])
    def test_kernel_acquisition(self, tmp_path: Path, mission: str) -> None:

        manager = io.SpiceKernelManager(mission, tmp_path)
        metakernel = manager.ensure_metakernel()

        # Modify metakernel to make test shorter
        metakernel_content = metakernel.read_text()
        counter: int = 0
        limit: int = 2
        with metakernel.open("w") as f:

            for line in metakernel_content.splitlines():

                if line.strip().startswith("'$KERNELS/"):
                    if counter >= limit:
                        continue
                    counter += 1
                f.write(line + "\n")

        # Ensure that the file was modified correctly
        kernel_list = manager._SpiceKernelManager__get_required_kernels(metakernel)  # type: ignore
        assert len(kernel_list) == limit

        # Download kernels
        manager.ensure_kernels(metakernel)

        # Check that the kernels were downloaded
        for kernel in kernel_list:
            assert (manager.kernels_folder / kernel).exists()


class TestVexInterface:

    vex_file = Path(__file__).parent / "data/GR035.vix"

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

    def test_load_stations_section(self) -> None:

        vex = io.Vex(self.vex_file)

        # Check that incorrect VEX with multiple IDs for same station fails
        # The stations also have different names, but they should be normalized
        # and then the ID will be duplicated
        wrong_vex = io.Vex(self.vex_file.parent / "mock.vix")
        with pytest.raises(SystemExit) as error:
            wrong_vex.load_station_ids_and_names()

        # No stations are ignored
        all_stations = vex.load_station_ids_and_names()
        assert len(all_stations) == 29
        assert all_stations["Ww"] == "WARK12M"

        # Some stations are ignored
        some_stations = vex.load_station_ids_and_names(["Cd"])
        assert len(some_stations) == 28
        assert "Cd" not in some_stations
