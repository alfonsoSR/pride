from pride.experiment import Baseline, Observation
from pride.source import NearFieldSource
from pride.station import Station
from pride.types import Band
import datetime
from astropy import time
import pytest
import numpy as np


def test_tstamps_handling_in_baseline_constructor() -> None:
    """Test handling of time stamps in Baseline constructor

    The constructor of Baseline takes a list of observations associated with a given station. Each `Observation` object includes a `time.Time` object with a list of time stamps from all the scans associated with observations performed from a given station. These time stamps are in UTC at the instantaneous, tectonic-corrected position of the station.

    The `tstamps` attribute of a `Baseline` object is a `time.Time` object containing the time stamps of all the observations performed from that baseline, in UTC at the tectonic-corrected location of the station, and sorted from first to last. When multiple observations are available for a given baseline, combining and sorting their time stamps requires first removing the location information from each `observation.tstamps` object, then merging them all into a single `time.Time`, and then updating this last object with the location information.

    In version 1.2.11, we found an issue (#31) with this process. The code would only work when a baseline had multiple observations, and fail otherwise. Having fixed this, this test is meant to ensure that the Baseline constructor works regardless of whether the baseline has one or many observations.
    """

    sources = [NearFieldSource("source_a"), NearFieldSource("source_b")]

    __station = Station("CEDUNA", "Cd", False, None)
    __center = Station("GEOCENTR", "00", True, None)

    __band = Band.__new__(Band)

    iso_epochs = [
        [
            "2013-12-29 10:11:12",
            "2013-12-29 10:11:13",
            "2013-12-29 10:11:11",
        ],
        [
            "2013-12-28 10:11:12",
            "2013-12-28 10:11:11",
            "2013-12-28 10:11:13",
        ],
    ]
    iso_epochs_combined_sorted = [
        "2013-12-28 10:11:11",
        "2013-12-28 10:11:12",
        "2013-12-28 10:11:13",
        "2013-12-29 10:11:11",
        "2013-12-29 10:11:12",
        "2013-12-29 10:11:13",
    ]

    tstamps_subsets = [
        [datetime.datetime.fromisoformat(epoch) for epoch in subset]
        for subset in iso_epochs
    ]
    tstamps_combined = [
        datetime.datetime.fromisoformat(epoch)
        for epoch in (iso_epochs[0] + iso_epochs[1])
    ]
    tstamps_combined_sorted = [
        datetime.datetime.fromisoformat(epoch)
        for epoch in iso_epochs_combined_sorted
    ]

    observation_subsets = [
        Observation(__station, source, __band, tstamps)
        for (tstamps, source) in zip(tstamps_subsets, sources)
    ]
    observations_combined = Observation(
        __station, sources[0], __band, tstamps_combined
    )

    baseline_multiple = Baseline(__center, __station, observation_subsets)
    baseline_combined = Baseline(__center, __station, [observations_combined])

    __expected = time.Time(tstamps_combined_sorted, scale="utc")
    expected = time.Time(
        tstamps_combined_sorted,
        scale="utc",
        location=__station.tectonic_corrected_location(__expected),
    )

    diff_multiple = (baseline_multiple.tstamps - expected).to_value("s")
    assert isinstance(diff_multiple, np.ndarray)
    for item in diff_multiple:
        assert item == pytest.approx(0)

    diff_combined = (baseline_combined.tstamps - expected).to_value("s")
    assert isinstance(diff_combined, np.ndarray)
    for item in diff_combined:
        assert item == pytest.approx(0)

    diff_combined_multiple = (
        baseline_multiple.tstamps - baseline_combined.tstamps
    ).to_value("s")
    assert isinstance(diff_combined_multiple, np.ndarray)
    for item in diff_combined_multiple:
        assert item == pytest.approx(0)

    return None
