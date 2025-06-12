import pytest
from pride.io.ionex.download import (
    get_ionex_path_for_date,
    download_compressed_ionex_file,
    decompress_ionex_file,
    download_ionex_file_for_date,
)
from pride.io.ionex.interface import IonexInterface
from astropy import time
from pathlib import Path
from pride.logger import log
import os
from scipy import interpolate
import numpy as np

# Path to directory with test data
DATA_DIRECTORY: Path = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize(
    ["epoch", "expected_path"],
    [
        (
            time.Time("2022-11-27T00:00:00", scale="utc"),
            "2022/331/IGS0OPSFIN_20223310000_01D_02H_GIM.INX.gz",
        ),
        (
            time.Time("2022-11-26T00:00:00", scale="utc"),
            "2022/330/igsg3300.22i.Z",
        ),
    ],
    ids=[
        "New format: date",
        "Old format: date",
    ],
)
def test_ionex_file_name(epoch: "time.Time", expected_path: str) -> None:

    ionex_path = get_ionex_path_for_date(epoch)
    assert ionex_path == expected_path

    return None


@pytest.mark.parametrize(
    ["epoch", "expected_path", "date_in_second_line"],
    [
        (
            time.Time("2022-11-27T00:00:00", scale="utc"),
            "IGS0OPSFIN_20223310000_01D_02H_GIM.INX.gz",
            "19-dec-22 17:58",
        ),
        (
            time.Time("2022-11-26T00:00:00", scale="utc"),
            "igsg3300.22i.Z",
            "6-dec-22 16:26",
        ),
    ],
    ids=[
        "New format",
        "Old format",
    ],
)
def test_download_and_decompress(
    epoch: "time.Time",
    expected_path: str,
    date_in_second_line: str,
    tmp_path: Path,
) -> None:

    # Define the output directory
    output_directory = tmp_path / "ionex_files"

    # Get the path to the IONEX file for the given date
    ionex_path = get_ionex_path_for_date(epoch)

    # Download the compressed IONEX file
    compressed_file = download_compressed_ionex_file(
        ionex_path, output_directory
    )
    assert compressed_file.exists()
    assert compressed_file.name == expected_path

    # Decompress the downloaded file
    decompressed_file = decompress_ionex_file(compressed_file)
    assert decompressed_file.exists()
    assert decompressed_file.name == str(Path(expected_path).with_suffix(""))
    assert not compressed_file.exists()

    # Get Unix timestamp for the last modification time
    creation_time = os.path.getmtime(decompressed_file)

    # Check content of decompressed file
    with decompressed_file.open("r") as f:
        lines = f.readlines()
        assert len(lines) > 1
        assert date_in_second_line in lines[1]

    new_decompressed_file = download_ionex_file_for_date(
        epoch, output_directory
    )
    assert new_decompressed_file.exists()
    assert creation_time == os.path.getmtime(new_decompressed_file)

    return None


@pytest.mark.parametrize(
    [
        "ionex_file",
        "expected_height_and_rearth",
        "expected_number_of_elements",
        "expected_start_and_end",
        "expected_epoch_lat_lon_tec_combinations",
    ],
    [
        (
            "ionex_files/new_ionex.INX",
            (450.0, 6371.0),
            13,
            (
                time.Time("2023-10-19T00:00:00"),
                time.Time("2023-10-20T00:00:00"),
            ),
            [(time.Time("2023-10-19T02:00:00"), 75.0, -160, 275)],
        ),
        (
            "ionex_files/old_ionex.13i",
            (450.0, 6371.0),
            13,
            (
                time.Time("2013-12-28T00:00:00"),
                time.Time("2013-12-29T00:00:00"),
            ),
            [(time.Time("2013-12-28T06:00:00"), -25.0, 150, 527)],
        ),
    ],
)
def test_read_tec_maps(
    ionex_file: str,
    expected_height_and_rearth: tuple[float, float],
    expected_number_of_elements: int,
    expected_start_and_end: tuple[time.Time, time.Time],
    expected_epoch_lat_lon_tec_combinations: list[
        tuple[time.Time, float, float, float]
    ],
) -> None:

    # Initialize ionex interface for chosen file
    ionex_path = DATA_DIRECTORY / ionex_file
    ionex = IonexInterface(ionex_path)

    # Read data from file
    tec_maps, ref_height, ref_rearth = ionex.read_data_from_ionex_file()

    # Check expected height and Earth radius
    assert (ref_height, ref_rearth) == expected_height_and_rearth

    # Check it is a dictionary with time.Time keys and interpolators as values
    assert isinstance(tec_maps, dict)
    for key, val in tec_maps.items():
        assert isinstance(key, time.Time)
        assert isinstance(val, interpolate.RegularGridInterpolator)

    # Ensure that the dictionary has the expected number of elements
    assert len(tec_maps.keys()) == expected_number_of_elements

    # Check initial and final epochs
    tec_epochs = list(tec_maps.keys())
    assert tec_epochs[0] == expected_start_and_end[0]
    assert tec_epochs[-1] == expected_start_and_end[1]

    # Check some TEC values
    for epoch, lat, lon, tec in expected_epoch_lat_lon_tec_combinations:
        assert tec_maps[epoch]([lon, lat]) == tec

    return None


# @pytest.mark.parametrize(
#     [
#         "epoch",
#         "expected_radius",
#         "expected_height",
#         "expected_tec_maps",
#         "content_validation_tuple",
#     ],
#     [
#         (
#             time.Time("2022-11-27T00:00:00", scale="utc"),
#             6371.0,
#             450.0,
#             13,
#             (-175.0, 87.5, 82),
#         ),
#         (
#             time.Time("2022-11-26T00:00:00", scale="utc"),
#             6371.0,
#             450.0,
#             13,
#             (-175.0, 87.5, 69),
#         ),
#     ],
#     ids=[
#         "New format",
#         "Old format",
#     ],
# )
# def test_ionex_interface(
#     epoch: "time.Time",
#     expected_height: float,
#     expected_radius: float,
#     expected_tec_maps: int,
#     content_validation_tuple: tuple[float, float, int],
#     tmp_path: Path,
# ) -> None:

#     # Directory with IONEX files from tests/data
#     ionex_path = DATA_DIRECTORY / ionex_file

#     # Initialize interface for file
#     ionex_file = download_ionex_file_for_date(epoch, tmp_path)
#     ionex = IonexInterface(ionex_file)

#     tec_maps, ref_height, ref_rearth = ionex.read_data_from_ionex_file()
#     assert isinstance(tec_maps, list)
#     assert isinstance(tec_maps[0], interpolate.RegularGridInterpolator)
#     assert isinstance(ref_height, float)
#     assert isinstance(ref_rearth, float)
#     assert ref_rearth == expected_radius
#     assert ref_height == expected_height
#     assert len(tec_maps) == expected_tec_maps

#     lon, lat, expected_tec = content_validation_tuple
#     assert tec_maps[0]([lon, lat]) == expected_tec

#     return None
