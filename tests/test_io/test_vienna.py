from pride.io.vienna.download import get_v3gr_url_for_epoch, download_v3gr_file
from pride.io.vienna.interface import V3GRInterface
import pytest
from astropy import time
from pathlib import Path
from pride import io
import os


@pytest.mark.parametrize(
    ["epoch", "expected_url"],
    [
        (
            time.Time("2022-11-27T00:00:00", scale="utc"),
            (
                "https://vmf.geo.tuwien.ac.at/trop_products/VLBI/V3GR/"
                "V3GR_OP/daily/2022/2022331.v3gr_r"
            ),
        ),
        (
            time.Time("2022-11-27T22:14:30", scale="utc"),
            (
                "https://vmf.geo.tuwien.ac.at/trop_products/VLBI/V3GR/"
                "V3GR_OP/daily/2022/2022331.v3gr_r"
            ),
        ),
    ],
    ids=[
        "Date",
        "Epoch",
    ],
)
def test_v3gr_url_for_epoch(epoch: "time.Time", expected_url: str) -> None:

    v3gr_url = get_v3gr_url_for_epoch(epoch)
    assert v3gr_url == expected_url

    return None


@pytest.mark.parametrize(
    ["epoch", "fails", "expected_third_line"],
    [
        (
            time.Time("2022-11-27T00:00:00", scale="utc"),
            False,
            (
                "ALGOPARK  59910.00  0.00122958  0.00063003  2.2233  0.0747"
                "   975.20   7.33   5.43  -0.661  -0.008   0.053  -0.015"
            ),
        ),
        (
            time.Time("2022-11-27T22:15:30", scale="utc"),
            False,
            (
                "ALGOPARK  59910.00  0.00122958  0.00063003  2.2233  0.0747"
                "   975.20   7.33   5.43  -0.661  -0.008   0.053  -0.015"
            ),
        ),
    ],
    ids=[
        "Date",
        "Epoch",
    ],
)
def test_download_v3gr_file(
    epoch: "time.Time", fails: bool, expected_third_line: str, tmp_path: Path
) -> None:

    # Get URL for the given epoch
    v3gr_url = get_v3gr_url_for_epoch(epoch)

    if fails:
        with pytest.raises(SystemExit):
            _ = download_v3gr_file(v3gr_url, tmp_path)
        return None

    # Download the V3GR file
    v3gr_file = download_v3gr_file(v3gr_url, tmp_path)
    assert v3gr_file.exists()
    assert v3gr_file.name == Path(v3gr_url).name
    assert v3gr_file.parent == tmp_path
    creation_time = os.path.getmtime(v3gr_file)

    content = v3gr_file.read_text().splitlines()
    assert content[2] == expected_third_line

    # Check for duplicated download
    new_v3gr_file = io.download_v3gr_file_for_epoch(epoch, tmp_path)
    assert new_v3gr_file.exists()
    assert creation_time == os.path.getmtime(new_v3gr_file)

    return None


@pytest.mark.parametrize(
    ["epoch", "station_name", "expected_coefficients", "fails"],
    [
        (
            time.Time("2013-12-28T00:00:00", scale="utc"),
            "ARECIBO",
            [
                56654.0,
                1.25957e-3,
                5.669e-4,
                2.1972,
                0.2062,
                -0.119,
                -0.050,
                -0.229,
                0.012,
            ],
            False,
        ),
        (
            time.Time("2022-11-27T00:00:00", scale="utc"),
            "INVALID",
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            True,
        ),
    ],
    ids=[
        "Valid station",
        "Invalid station",
    ],
)
def test_v3gr_interface(
    epoch: "time.Time",
    station_name: str,
    expected_coefficients: list[float],
    fails: bool,
    tmp_path: Path,
) -> None:

    # Initialize the V3GR interface
    v3gr_file = io.download_v3gr_file_for_epoch(epoch, tmp_path)
    v3gr_interface = V3GRInterface(v3gr_file)

    if fails:
        with pytest.raises(SystemExit):
            _ = v3gr_interface.read_v3gr_data_for_station(station_name)
        return None

    # Read V3GR data for the specified station
    coefficients = v3gr_interface.read_v3gr_data_for_station(station_name)
    assert coefficients == expected_coefficients
    assert len(coefficients) == 9

    return None
