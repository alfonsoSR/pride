from pride import utils, io
from pride.utils.misc import is_station_in_line
import datetime
import pytest
import numpy as np
from astropy import time
import struct

TOL = 1e-15


@pytest.mark.parametrize(
    [
        "offsets",
        "expected_length",
        "expected_step",
        "fails",
        "respects_min_obs",
    ],
    [
        ((0, 130), 14, 10.0, False, True),  # First condition - No offset
        ((10, 130), 13, 10.0, False, True),  # First condition - Offset
        ((0, 90), 11, 9.0, False, True),  # Second condition - No offset
        ((10, 100), 11, 9.0, False, True),  # Second condition - Offset
        ((0, 100), 11, 10.0, False, True),  # Second condition - Default step
        ((0, 20), 11, 2.0, False, True),  # Second condition - Min step
        ((0, 70), 11, 7.0, False, True),  # Second condition - Min nobs
        ((0, 15), 8, 15 / 7, False, False),  # Third condition - No offset
        ((20, 35), 8, 15 / 7, False, False),  # Third condition - Offset
        ((20, 10), 0, 0, True, True),  # Invalid: Wrong order
        ((-10, 0), 0, 0, True, True),  # Invalid: Negative initial offset
        ((-20, -10), 0, 0, True, True),  # Invalid: Negative final offset
    ],
    ids=[
        "First condition - No offset",
        "First condition - Offset",
        "Second condition - No offset",
        "Second condition - Offset",
        "Second condition - Default step",
        "Second condition - Min step",
        "Second condition - Min nobs",
        "Third condition - No offset",
        "Third condition - Offset",
        "Invalid: Wrong order",
        "Invalid: Negative initial offset",
        "Invalid: Negative final offset",
    ],
)
def test_scan_discretization(
    offsets: tuple[int, int],
    expected_length: int,
    expected_step: float,
    fails: bool,
    respects_min_obs: bool,
) -> None:

    # Reference epoch is irrelevant > Use same value for all tests
    reference_epoch = datetime.datetime(2000, 6, 28, 13, 12, 0)

    # Internal configuration
    internal_setup = io.load_catalog("config.yaml")["Configuration"]

    # Discretize scan
    if fails:
        with pytest.raises(SystemExit):
            tstamps = utils.discretize_scan(reference_epoch, *offsets)
        return None
    else:
        tstamps = utils.discretize_scan(reference_epoch, *offsets)

    # Calculate quantities for testing
    dt = np.array([(ti - reference_epoch).total_seconds() for ti in tstamps])
    nobs = len(tstamps)
    reference_step = dt[1] - dt[0]
    actual_steps = dt[1:] - dt[:-1]

    # Ensure that step is valid, expected, and all steps are equal
    assert np.all(np.isclose(actual_steps, reference_step, atol=TOL))
    assert reference_step >= internal_setup["min_scan_step"]
    assert np.isclose(reference_step, expected_step, atol=TOL)

    # Ensure that initial and final offsets are respected
    assert np.isclose(dt[0], offsets[0], atol=TOL)
    assert np.isclose(dt[-1], offsets[1], atol=TOL)

    # Ensure that the internal configuration is respected
    assert len(tstamps) == expected_length
    if respects_min_obs:
        assert nobs >= internal_setup["min_obs_per_scan"]

    return None


@pytest.mark.parametrize(
    ["epoch", "expected_date"],
    [
        (
            time.Time("2000-06-28T00:00:00", scale="utc"),
            time.Time("2000-06-28T00:00:00", scale="utc"),
        ),
        (
            time.Time("2000-06-28T14:07:00", scale="utc"),
            time.Time("2000-06-28T00:00:00", scale="utc"),
        ),
        (
            time.Time("2000-06-28T23:59:59.999", scale="utc"),
            time.Time("2000-06-28T00:00:00", scale="utc"),
        ),
        (
            time.Time("2000-06-28T00:00:00.001", scale="utc"),
            time.Time("2000-06-28T00:00:00", scale="utc"),
        ),
    ],
    ids=[
        "Epoch is date",
        "Non conflictive",
        "Close to midnight of next day",
        "Close to midnight of current day",
    ],
)
def test_date_from_epoch(
    epoch: "time.Time", expected_date: "time.Time"
) -> None:

    date = utils.get_date_from_epoch(epoch)
    difference = (date - expected_date).to_value("s")
    assert difference == 0

    return None


@pytest.mark.parametrize(
    ["date", "expected_week", "fails"],
    [
        (
            time.Time("2000-06-28T00:00:00", scale="utc"),
            1068,
            False,
        ),
        (
            time.Time("2000-06-28T00:04:00", scale="utc"),
            0,
            True,
        ),
        (
            time.Time("2000-06-28T00:00:00.001", scale="utc"),
            1068,
            False,
        ),
    ],
    ids=[
        "Valid date",
        "Invalid date",
        "Valid, possibly conflictive",
    ],
)
def test_gps_week_from_date(
    date: "time.Time", expected_week: int, fails: bool
) -> None:

    # Test if the function fails as expected
    if fails:
        with pytest.raises(SystemExit):
            _ = utils.get_gps_week_for_date(date)
        return None

    week = utils.get_gps_week_for_date(date)
    assert week == expected_week

    return None


@pytest.mark.parametrize(
    ["epoch", "expected_year", "expected_day_of_year", "expected_hour"],
    [
        (
            time.Time("2000-06-28T00:05:03", scale="utc"),
            2000,
            180,
            0,
        ),
        (
            time.Time("2000-06-28T13:12:05", scale="utc"),
            2000,
            180,
            13,
        ),
    ],
)
def test_small_time_utilities(
    epoch: "time.Time",
    expected_year: int,
    expected_day_of_year: int,
    expected_hour: int,
) -> None:

    assert utils.get_year_from_epoch(epoch) == expected_year
    assert utils.get_day_of_year_from_epoch(epoch) == expected_day_of_year
    assert utils.get_hour_from_epoch(epoch) == expected_hour


@pytest.mark.parametrize(
    ["station", "line", "found"],
    [
        (
            "HARTRAO",
            "HARTRAO xxxxxxxx xxxxxxxx xxxxxxx",
            True,
        ),  # Match beginning of line
        (
            "HARTRAO",
            "xxxxxxxx xxxxxxxx HARTRAO xxxxxxx",
            True,
        ),  # Match middle of line
        (
            "HARTRAO",
            "xxxxxxxx xxxxxxxx xxxxxxxx HARTRAO",
            True,
        ),  # Match end of line
        (
            "HARTRAO",
            "xxxxxxx xxxxxxxx xxxxxxxx xxxxxxx",
            False,
        ),  # No match
        (
            "HARTRAO",
            "HARTRAOX xxxxxxxx xxxxxxxx xxxxxxx",
            False,
        ),  # Partial match
        (
            "HARTRAO",
            "",
            False,
        ),  # Empty line
    ],
    ids=[
        "Match beginning of line",
        "Match middle of line",
        "Match end of line",
        "No match",
        "Partial match",
        "Empty line",
    ],
)
def test_is_station_in_line(station: str, line: str, found: bool) -> None:

    assert is_station_in_line(station, line) == found


@pytest.mark.parametrize(
    ["epoch", "is_date"],
    [
        (time.Time("2000-06-28T00:00:00"), True),
        (time.Time("2000-06-28T00:00:00.0000"), True),
        (time.Time("2000-06-28T00:00:00.009"), True),
        (time.Time("2000-06-28T00:00:00.01"), False),
        (time.Time("2000-06-28T00:00:01"), False),
    ],
)
def test_epoch_is_date(epoch: "time.Time", is_date: bool) -> None:

    assert utils.epoch_is_date(epoch) == is_date

    return None


@pytest.mark.parametrize(
    ["contents_format", "start", "expected", "expected_index", "fails"],
    [
        ("<ix", 0, [4], 5, False),  # Read integer
        ("<3sx", 5, ["AsR"], 9, False),  # Read strings
        ("<3d2x", 9, [1.0, 2.0, 3.0], 35, False),  # Read doubles (with padding)
        (
            "<3d",
            9,
            [1.0, 2.0, 3.0],
            33,
            False,
        ),  # Read doubles (without padding)
        ("<o", 0, [], 0, True),  # Fail on incorrect format
        ("<10x", 30, [], 0, True),  # Fail on buffer overflow
    ],
)
def test_peek_buffer(
    contents_format: str,
    start: int,
    expected: list,
    expected_index: int,
    fails: bool,
) -> None:

    common_buffer = struct.pack(
        "<ix3sx3d2x", 4, "AsR".encode("utf-8"), 1.0, 2.0, 3.0
    )

    # For tests of expected failures: should always raise BufferError
    if fails:
        with pytest.raises(BufferError):
            _ = utils.peek_buffer(common_buffer, contents_format, start)
        return None

    output, index = utils.peek_buffer(common_buffer, contents_format, start)

    assert output == expected
    assert index == expected_index

    return None
