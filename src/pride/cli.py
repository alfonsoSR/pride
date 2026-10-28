from .experiment import Experiment
from .doppler import DopplerPredictor
import argparse
from .logger import log


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

            baseline.update_station_with_geophysical_displacements(
                experiment.displacement_models,
                experiment.eops,
            )

            for observation in baseline.observations:

                # Calculate spherical coordinates of the source
                observation.update_with_source_coordinates()

                # Calculate delays for the observation
                observation.calculate_delays(experiment.delay_models)

                # Calculate UVW projections for the observation
                if experiment.setup.uvw["calculate"]:
                    observation.calculate_uvw(
                        experiment.setup.uvw["method"],
                        experiment.delay_models,
                    )

        # Calculate Doppler predictions for the near-field target
        doppler = None
        if experiment.setup.doppler["calculate"]:
            doppler = DopplerPredictor(experiment)
            doppler.run()

    experiment.save_output(doppler)

    return None
