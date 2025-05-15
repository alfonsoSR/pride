from pride.io.ramping_data import ramping_data as rd
import pytest
from astropy import time
from pathlib import Path
import numpy as np


@pytest.mark.parametrize(
    ["t0", "tend", "is_in", "should_stop"],
    [
        (
            "2000-06-28T12:14:24.000",
            "2000-06-28T17:17:22.000",
            True,
            False,
        ),  # Fully contained
        (
            "2000-06-28T12:14:22.000",
            "2000-06-28T17:17:22.000",
            True,
            False,
        ),  # Left out, right in
        (
            "2000-06-28T12:14:22.000",
            "2000-06-28T17:17:23.000",
            True,
            False,
        ),  # Left out, right in (exact)
        (
            "2000-06-28T12:14:24.000",
            "2000-06-28T17:17:24.000",
            True,
            False,
        ),  # Left in, right out
        (
            "2000-06-28T12:14:23.000",
            "2000-06-28T17:17:24.000",
            True,
            False,
        ),  # Left in (exact), right out
        (
            "2000-06-28T12:14:21.000",
            "2000-06-28T12:14:22.000",
            False,
            False,
        ),  # All out before
        (
            "2000-06-28T17:17:24.000",
            "2000-06-28T17:17:25.000",
            False,
            True,
        ),  # All out after
    ],
    ids=[
        "Fully contained",
        "Left out, right in",
        "Left out, right in (exact)",
        "Left in, right out",
        "Left in (exact), right out",
        "All out before",
        "All out after",
    ],
)
def test_data_is_in_time_interval(
    t0: str, tend: str, is_in: bool, should_stop: bool
) -> None:
    """Test function used to determine if data is in a certain interval"""

    # Reference time interval for all the tests
    time_interval = (
        time.Time("2000-06-28T12:14:23.000"),
        time.Time("2000-06-28T12:17:23.000"),
    )

    line_in_interval, is_after_interval = rd.__data_is_in_time_interval(
        t0, tend, time_interval
    )
    assert line_in_interval == is_in
    assert is_after_interval == should_stop

    return None


@pytest.mark.parametrize(
    [
        "ramping_type",
        "time_interval",
        "loads_data",
        "number_of_items_loaded",
        "first_line_loaded",
    ],
    [
        (
            "one-way",
            (
                time.Time("2008-01-05 03:21:43.000"),
                time.Time("2008-01-05 03:21:44.000"),
            ),
            True,
            11,
            (
                time.Time("2008-01-05 03:21:42.986000"),
                time.Time("2008-01-05 03:21:43.086000"),
                8.4190811645200033e09,
                4.6999999999999997e-05,
                None,
            ),
        ),
        (
            "three-way",
            (
                time.Time("2010-08-30 07:45:00"),
                time.Time("2011-03-25 07:40:05.000000"),
            ),
            True,
            4,
            (
                time.Time("2010-08-30 07:45:06.000000"),
                time.Time("2010-08-30 11:58:46.000000"),
                7.1654630080000000e09,
                0.0000000000000000e00,
                "CEBREROS",
            ),
        ),
    ],
)
def test_load_ramping_data(
    ramping_type: str,
    time_interval: tuple[time.Time, time.Time],
    loads_data: bool,
    number_of_items_loaded: int,
    first_line_loaded: tuple[time.Time, time.Time, float, float, str | None],
) -> None:

    # Path to ramping file
    ramping_file = Path(__file__).parent.parent / f"data/{ramping_type}.txt"

    # Load ramping data
    ramping_data = rd.load_ramping_data(
        ramping_file=ramping_file,
        ramping_type=ramping_type,  # type: ignore
        time_interval=time_interval,
    )

    # Check loaded data
    if not loads_data:
        assert ramping_data is None
        return None
    assert ramping_data is not None

    # Check the fields of the dictionary are correct
    assert isinstance(ramping_data["t0"], time.Time)
    assert isinstance(ramping_data["t1"], time.Time)
    assert isinstance(ramping_data["f0"], np.ndarray)
    assert isinstance(ramping_data["df"], np.ndarray)
    if ramping_type == "one-way":
        assert "uplink" not in ramping_data
    else:
        assert isinstance(ramping_data["uplink"], list)
        assert isinstance(ramping_data["uplink"][0], str)

    # Check that all the entries of the dictionary are the same length
    assert len(ramping_data["t0"]) == number_of_items_loaded
    assert len(ramping_data["t1"]) == number_of_items_loaded
    assert len(ramping_data["f0"]) == number_of_items_loaded
    assert len(ramping_data["df"]) == number_of_items_loaded
    if ramping_type == "three-way":
        assert len(ramping_data["uplink"]) == number_of_items_loaded

    # Check the that the first items loaded are correct
    assert ramping_data["t0"][0] == first_line_loaded[0]
    assert ramping_data["t1"][0] == first_line_loaded[1]
    assert ramping_data["f0"][0] == first_line_loaded[2]
    assert ramping_data["df"][0] == first_line_loaded[3]
    if ramping_type == "three-way":
        assert ramping_data["uplink"][0] == first_line_loaded[4]

    return None
