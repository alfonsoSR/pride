from pride import io, types
from pathlib import Path
import os
import pytest
from astropy import time
from datetime import datetime
import numpy as np
from pride.io.vex.interface import ScanData


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
