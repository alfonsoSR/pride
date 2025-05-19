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
        "epoch",
        "expected_radius",
        "expected_height",
        "expected_tec_maps",
        "content_validation_tuple",
    ],
    [
        (
            time.Time("2022-11-27T00:00:00", scale="utc"),
            6371.0,
            450.0,
            13,
            (-175.0, 87.5, 82),
        ),
        (
            time.Time("2022-11-26T00:00:00", scale="utc"),
            6371.0,
            450.0,
            13,
            (-175.0, 87.5, 69),
        ),
    ],
    ids=[
        "New format",
        "Old format",
    ],
)
def test_ionex_interface(
    epoch: "time.Time",
    expected_height: float,
    expected_radius: float,
    expected_tec_maps: int,
    content_validation_tuple: tuple[float, float, int],
    tmp_path: Path,
) -> None:

    # Initialize interface for file
    ionex_file = download_ionex_file_for_date(epoch, tmp_path)
    ionex = IonexInterface(ionex_file)

    tec_maps, ref_height, ref_rearth = ionex.read_data_from_ionex_file()
    assert isinstance(tec_maps, list)
    assert isinstance(tec_maps[0], interpolate.RegularGridInterpolator)
    assert isinstance(ref_height, float)
    assert isinstance(ref_rearth, float)
    assert ref_rearth == expected_radius
    assert ref_height == expected_height
    assert len(tec_maps) == expected_tec_maps

    lon, lat, expected_tec = content_validation_tuple
    assert tec_maps[0]([lon, lat]) == expected_tec

    return None
