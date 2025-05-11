from pride import utils, io
import datetime
import pytest
import numpy as np

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
