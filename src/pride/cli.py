from .experiment import Experiment
import argparse


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
            baseline.update_with_observations()

            # Update station coordinates with geophysical displacements
            baseline.update_station_with_geophysical_displacements()

            for observation in baseline.observations:

                # Calculate spherical coordinates of the source
                observation.update_with_source_coordinates()

                # Calculate delays for the observation
                observation.calculate_delays()

    # Save the output
    experiment.save_output()

    return None
