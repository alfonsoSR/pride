from pride.experiment import Experiment
from pride.logger import log

log.setLevel("DEBUG")

experiment = Experiment("config.yaml")

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

experiment.save_output()
