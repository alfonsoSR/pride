from .experiment import Experiment
import argparse
from . import coordinates as coord
from .logger import log
from astropy import time
import numpy as np


class PrideArgumentParser(argparse.ArgumentParser):

    def __init__(self) -> None:

        # Default initialization
        super().__init__(
            prog="pride", description="Estimate delays for VLBI experiment"
        )

        # Configuration file
        self.add_argument(
            "-c",
            "--config-file",
            dest="configuration_file",
            help="Path to configuration file",
            required=True,
        )

        return None


def process_experiment() -> None:

    args = PrideArgumentParser().parse_args()
    configuration_file = args.configuration_file

    # Initialize experiment
    experiment = Experiment(configuration_file)

    with experiment.spice_kernels():

        for baseline in experiment.baselines:

            # Update baseline with data from observations
            # Augment time stamps with +/- 1 second around each epoch
            __augmented_tstamps: time.Time = (
                baseline.tstamps[:, None]
                + time.TimeDelta([-1, 0, 1], format="sec")[None, :]
            ).ravel()  # type: ignore
            a_tstamps = time.Time(
                __augmented_tstamps,
                scale="utc",
                location=baseline.station.tectonic_corrected_location(
                    __augmented_tstamps
                ),
            )
            assert a_tstamps.location is not None

            # Calculate geodetic coordinates of station
            a_geodetic = a_tstamps.location.to_geodetic("GRS80")
            a_lat = np.array(a_geodetic.lat.rad, dtype=float)
            a_lon = np.array(a_geodetic.lon.rad, dtype=float)

            # Calculate rotation matrices
            augmented_eops = experiment.eops.at_epoch(a_tstamps, unit="arcsec")
            # self.a_eops = eops.at_epoch(self.a_tstamps, unit="arcsec")
            a_icrf2itrf = coord.icrf2itrf(augmented_eops, a_tstamps)
            a_seu2itrf = coord.seu2itrf(a_lat, a_lon)

            # Calculate derivative of ICRF to ITRF rotation matrix
            dt_tdb: np.ndarray = (
                (a_tstamps.tdb[2::3] - a_tstamps.tdb[0::3])
                .to("s")  # type: ignore
                .value
            )
            diff_icrf2itrf = a_icrf2itrf[2::3] - a_icrf2itrf[0::3]
            dot_icrf2itrf = diff_icrf2itrf / dt_tdb[:, None, None]

            # Get attributes at observation epochs
            icrf2itrf = a_icrf2itrf[1::3]
            seu2itrf = a_seu2itrf[1::3]

            # Update observations with calculated data
            for observation in baseline.observations:

                # Get the index of the time stamps associated with the observation
                flags = np.sum(
                    observation.tstamps[:, None] == baseline.tstamps[None, :],
                    axis=0,
                    dtype=bool,
                )

                observation.icrf2itrf = icrf2itrf[flags]
                observation.seu2itrf = seu2itrf[flags]
                observation.dot_icrf2itrf = dot_icrf2itrf[flags]

            # Update station coordinates with geophysical displacements
            baseline.update_station_with_geophysical_displacements(
                experiment.displacement_models,
                a_tstamps,
                augmented_eops,
                a_icrf2itrf,
                a_seu2itrf,
                a_lat,
                a_lon,
                icrf2itrf,
                dot_icrf2itrf,
            )

            for observation in baseline.observations:

                # Calculate spherical coordinates of the source
                observation.update_with_source_coordinates()

                # Calculate delays for the observation
                observation.calculate_delays(experiment.delay_models)

                # Calculate Doppler for the observation
                observation.calculate_doppler(experiment.doppler_models)

    experiment.save_output()

    return None
