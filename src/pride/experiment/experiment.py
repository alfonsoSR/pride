from typing import TYPE_CHECKING, Generator
from pathlib import Path
from contextlib import contextmanager
import spiceypy as spice
from .. import io
from ..logger import log
from astropy import time
import datetime
from ..types import Band
from ..coordinates import EOP
from ..displacements import DISPLACEMENT_MODELS
from ..delays import DELAY_MODELS
from ..source import Source, NearFieldSource, FarFieldSource
from .station import Station
from .baseline import Baseline
from .observation import Observation
import numpy as np
from .. import utils
from ..io.vex.interface import VEX_DATE_FORMAT

if TYPE_CHECKING:
    from ..displacements.core import Displacement
    from ..delays.core import Delay


class Experiment:
    """VLBI experiment"""

    def __init__(self, setup: str | Path) -> None:

        # Load setup from configuration file
        _setup_path = Path(setup).absolute()
        if not _setup_path.is_file():
            log.error(
                f"Failed to initialize {self.name} experiment: "
                f"Configuration file {_setup_path} not found"
            )
            exit(1)
        self.setup = io.Setup(str(_setup_path))

        # Generate interface to VEX file
        _vex_path = _setup_path.parent / self.setup.general["vex"]
        self.__vex = io.Vex(_vex_path)

        # Experiment metadata
        self.name = self.__vex.experiment_name
        self.initial_epoch = self.__vex.experiment_start
        self.final_epoch = self.__vex.experiment_stop
        log.info(f"Initializing {self.name} experiment")

        # Load observation modes from VEX file
        self.modes = self.__vex.load_observation_modes()

        # Load clock parameters from VEX file
        self.clock_parameters = self.__vex.load_clock_parameters()
        self.clock_offsets = self.clock_parameters  # Backwards compatibility

        # EOPs: For transformations between ITRF and ICRF
        self.eops = EOP.from_experiment(self)

        # Load target information
        self.target = io.get_target_information(self.setup.general["target"])

        # Ensure SPICE kernels for target
        self.metakernel = self.__ensure_spice_kernels(self.target["short_name"])

        # Load sources
        self.sources = self.load_sources(self.__vex)

        # Define phase center
        if self.setup.general["phase_center"] != "GEOCENTR":
            log.error(
                f"Failed to initialize {self.name} experiment: "
                "Using a station as phase center is currently not supported"
            )
            exit(1)
        self.phase_center = Station.from_experiment(
            self.setup.general["phase_center"], "00", self
        )

        # Initialize baselines
        self.baselines = self.initialize_baselines(
            self.__vex, self.setup.general["ignore_stations"]
        )

        # Initialize delay and displacement models
        self.requires_spice = False
        self.displacement_models = self.initialize_displacement_models()
        self.delay_models = self.initialize_delay_models()

        return None

    def __ensure_spice_kernels(self, target: str) -> Path:
        """Ensure SPICE kernels are available for the target

        :param target: Name of the target
        :return: Path to the metakernel
        """

        # Initialize SPICE kernel manager
        kernel_manager = io.SpiceKernelManager(
            mission=target,
            kernels_folder=self.setup.resources["ephemerides"],
        )

        # Ensure metakernel
        metakernel = kernel_manager.ensure_metakernel()

        # Ensure SPICE kernels listed in the metakernel
        kernel_manager.ensure_kernels(metakernel)

        return metakernel

    def load_sources(self, vex: "io.Vex") -> dict[str, "Source"]:
        """Load sources from VEX file

        Parses the SOURCE section of the VEX file and retrieves the name, type and coordinates of all the sources involved in the experiment. NearFieldSource and FarFieldSource objects are initialized for each source based on the 'source_type' read from the VEX, with the program raising an error if this attribute is not set.

        :param vex: Interface to VEX file
        :return: Dictionary with source name as key and Source object as value.
        """

        sources: dict[str, "Source"] = {}

        # Initialize far field sources from VEX file
        for source_name in vex.experiment_source_names:

            # Load source data from VEX file
            source_data = vex.load_source_data(source_name)

            # Initialize FarFieldSource objects for calibrators
            if source_data.source_type == "calibrator":

                assert source_data.right_ascension is not None  # Sanity
                assert source_data.declination is not None  # Sanity

                sources[source_name] = FarFieldSource(
                    source_data.name,
                    source_data.right_ascension,
                    source_data.declination,
                )

        # Initialize near field source
        sources[self.target["short_name"]] = NearFieldSource.from_experiment(
            self
        )

        return sources

    def initialize_displacement_models(self) -> list["Displacement"]:
        """Initialize displacement models

        Iterates over the 'Displacements' section of the configuration file and, for each of the names set to 'true', it looks for an equally named class in the 'DISPLACEMENT_MODELS' dictionary. If the class is not found, an error is raised indicating that a requested displacement is not available, otherwise, the displacement is initialized with the experiment. Initialization involves calling the 'ensure_resources' method of the displacement object.

        :return: List of 'empty' (no resources loaded) displacement objects
        """

        log.info("Initializing displacement models")

        displacement_models: list["Displacement"] = []
        for displacement, calculate in self.setup.displacements.items():
            if calculate:
                if displacement not in DISPLACEMENT_MODELS:
                    log.error(
                        f"Failed to initialize {displacement} displacement: "
                        "Model not found"
                    )
                    exit(1)
                displacement_models.append(DISPLACEMENT_MODELS[displacement]())
                if DISPLACEMENT_MODELS[displacement].requires_spice:
                    self.requires_spice = True

        log.info("Displacement models successfully initialized")

        return displacement_models

    def initialize_delay_models(self) -> list["Delay"]:
        """Initialize delay models

        Iterates over the 'Delays' section of the configuration file and, for each value with 'calculate' set to 'true', it looks for an equally named class in the 'DELAY_MODELS' dictionary. If the class is not found, an error is raised indicating that a requested delay is not available, otherwise, the delay is initialized with the experiment. Initialization involves calling the 'ensure_resources' and 'load_resources' methods of the delay object.

        :return: List of delay objects equiped with resources
        """

        log.info("Initializing delay models")

        _delay_models: list["Delay"] = []
        for delay_id, delay_config in self.setup.delays.items():
            if delay_config["calculate"]:
                if delay_id not in DELAY_MODELS:
                    log.error(
                        f"Failed to initialize {delay_id} delay: Model not found"
                    )
                    exit(1)
                _delay_models.append(DELAY_MODELS[delay_id](self))
                if DELAY_MODELS[delay_id].requires_spice:
                    self.requires_spice = True

        log.info("Delay models successfully initialized")

        return _delay_models

    def collect_observation_bands_and_timestamps(self, vex: "io.Vex") -> tuple[
        dict[str, dict[str, "Band"]],
        dict[str, dict[str, list[datetime.datetime]]],
    ]:
        """Group scan data into observations

        An Observation object collects all the scans of a source from a given
        station. This function goes over the SCHED section of the VEX file and
        groups the scan information per station and source. The output of this
        function is the input for observation initialization.

        :param vex: Interface to VEX file
        :return observation_bands: Two-level dictionary containing the band
        of each observation, indexed by station and source.
        :return observation_tstamps: Two-level dictionary containing the
        timestamps of each observation, indexed by station and source.
        """

        # Initialize output dictionaries
        observation_bands: dict[str, dict[str, "Band"]] = {}
        observation_tstamps: dict[str, dict[str, list[datetime.datetime]]] = {}

        # Parse VEX to identify cases of zero-gap scans
        zero_gap_scans = io.vex.get_list_of_zero_gap_scans(
            vex, self.target["short_name"]
        )

        # Loop over all scans in the VEX file
        for scan_id in vex.experiment_scans_ids:

            # Load scan data
            scan_data = vex.load_single_scan_data(
                scan_id, self.target["short_name"]
            )

            # Loop over all the stations involved in the scan
            for station_id, (
                initial_offset,
                final_offset,
            ) in scan_data.offsets_per_station.items():

                # If scan is zero-gap for this station, add 1 second offset
                if (scan_id, station_id) in zero_gap_scans:
                    log.warning(
                        f"Detected zero-gap for station {station_id} in scan "
                        f"{scan_id}: Applying additional offset of one second"
                    )
                    initial_offset += 1

                # Calculate timestamps for this scan
                scan_timestamps = utils.discretize_scan(
                    scan_data.initial_epoch,
                    initial_offset,
                    final_offset,
                    scan_id,
                )

                # If station is not in dictionaries, add it
                if station_id not in observation_bands:
                    observation_bands[station_id] = {}
                    observation_tstamps[station_id] = {}

                # Get observation band for this station
                band: "Band" = self.modes[
                    scan_data.observation_mode
                ].get_station_band(station_id)

                # Add observation data to dictionaries
                source = scan_data.source_name  # To make the code more readable
                if source not in observation_bands[station_id]:
                    observation_bands[station_id][source] = band
                    observation_tstamps[station_id][source] = scan_timestamps
                else:
                    assert observation_bands[station_id][source] == band
                    observation_tstamps[station_id][source] += scan_timestamps

        return observation_bands, observation_tstamps

    def initialize_baselines(
        self, vex: "io.Vex", ignored_stations: list[str]
    ) -> list["Baseline"]:
        """Initialize baselines

        The function creates an empty Baseline object for each station involved
        in the experiment, and then populates them with all the observations
        performed from that station.

        Information about observations is obtained by grouping the data of the
        scans listed in the VEX file per station-source pair. Check the
        __collect_observation_bands_and_timestamps function for more details.

        :param vex: Interface to VEX file
        :param ignored_stations: List of stations to ignore
        :return baselines: List of Baseline objects populated with observations.
        """

        log.info("Initializing baselines")

        # Load station IDs and names from VEX file
        stations_dictionary = vex.load_station_ids_and_names(ignored_stations)

        # Initialize dictionary of empty baselines (without observations)
        log.info("Initializing empty baselines")
        baselines_dictionary: dict[str, "Baseline"] = {
            station_id: Baseline(
                center=self.phase_center,
                station=Station.from_experiment(station_name, station_id, self),
            )
            for station_id, station_name in stations_dictionary.items()
        }

        # Collect observation bands and timestamps
        observation_bands, observation_tstamps = (
            self.collect_observation_bands_and_timestamps(vex)
        )

        # Update baselines with observations
        log.info("Updating baselines with observations")
        for baseline_id in observation_bands:
            for source_id in observation_bands[baseline_id]:

                # Create observation object
                _observation = Observation.from_experiment(
                    baselines_dictionary[baseline_id],
                    self.sources[source_id],
                    observation_bands[baseline_id][source_id],
                    observation_tstamps[baseline_id][source_id],
                    self,
                )

                # Update baseline with observation
                baselines_dictionary[baseline_id].add_observation(_observation)

        return list(baselines_dictionary.values())

    def load_clock_offsets(self):
        """Load clock offset data from VEX"""

        log.warning("load_clock_offsets is deprecated")

        return self.__vex.load_clock_parameters()

    @contextmanager
    def spice_kernels(self) -> Generator:
        """Context manager to load SPICE kernels"""

        try:
            log.debug("Loaded SPICE kernels")
            spice.furnsh(str(self.metakernel))

            yield None
        finally:
            log.debug("Unloaded SPICE kernels")
            spice.kclear()

        return None

    def save_output(self) -> None:
        log.warning("The save_output method should be refactored!")

        # Initialize output directory
        outdir = Path(self.setup.general["output_directory"]).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        # Output files and observations
        observations: dict[tuple[str, str], "Observation"] = {}
        output_files: dict[str, "io.DelFileGenerator"] = {}
        for baseline in self.baselines:

            # Station code
            station_id = baseline.station.id.title()

            # Initialize output file for observation
            output_files[station_id] = io.DelFileGenerator(
                outdir / f"{self.name}_{station_id}.del"
            )
            output_files[station_id].add_header(station_id)

            # Add observations to dictionary
            for observation in baseline.observations:
                observations[(station_id, observation.source.name)] = (
                    observation
                )

        # Main loop
        sorted_scans = sorted([s for s in self.__vex.experiment_scans_ids])
        # sorted_scans = sorted([s for s in self.vex["SCHED"]])
        for scan_id in sorted_scans:

            _scan_data = self.__vex.load_single_scan_data(
                scan_id, self.target["short_name"]
            )

            _stations = list(_scan_data.offsets_per_station.keys())

            # IDs of stations involved in scan
            scan_data = self.__vex._Vex__content["SCHED"][scan_id].getall(
                "station"
            )
            scan_stations = [s[0] for s in scan_data]
            assert scan_stations == _stations  # Sanity check

            # VEX-file and internal ID of source
            _scan_sources = [
                self.__vex._Vex__content["SCHED"][scan_id]["source"]
            ]
            if len(_scan_sources) != 1:
                log.error(f"Scan {scan_id} has multiple sources")
                exit(1)
            scan_source_id = _scan_sources[0]  # ID from VEX file

            _source_data = self.__vex._Vex__content["SOURCE"][scan_source_id]
            if _source_data["source_type"] == "target":
                scan_source = self.target["short_name"]  # Internal ID
            else:
                scan_source = scan_source_id  # Internal ID
            assert scan_source == _scan_data.source_name  # Sanity check

            # Loop over stations in scan
            for station_id in scan_stations:

                # Get observation for this scan
                assert (station_id, scan_source) in observations  # Sanity
                observation = observations[(station_id, scan_source)]

                # Initial and final epochs for scan
                t0 = time.Time.strptime(
                    self.__vex._Vex__content["SCHED"][scan_id]["start"],
                    VEX_DATE_FORMAT,
                )
                dt = time.TimeDelta(
                    int(scan_data[0][2].split()[0]), format="sec"
                )
                tend = t0 + dt

                # Timestamps and delays of current scan
                scan_mask = (observation.tstamps >= t0) & (
                    observation.tstamps <= tend
                )
                scan_tstamps = observation.tstamps[scan_mask]
                scan_delays = observation.delays[scan_mask]

                # Get integral part of MJD
                _mjd = np.array(scan_tstamps.mjd, dtype=int)  # type: ignore
                mjd1 = _mjd[0]

                # Fractional part of MJD in seconds
                mjd_diff = time.TimeDelta(_mjd - mjd1, format="jd")
                _mjd2 = time.TimeDelta(scan_tstamps.jd2 + 0.5, format="jd")  # type: ignore
                mjd2 = (mjd_diff + _mjd2).to("s").value  # type: ignore

                # Write data to output file
                # NOTE: Current version of the program does not include
                # Doppler shifts or UVW projections. These entries are set
                # to zero (to 1 for the amplitude of the Doppler shift)
                zero = np.zeros_like(mjd2)
                data = np.array(
                    [mjd2, zero, zero, zero, scan_delays, zero, zero + 1.0]
                ).T
                output_files[station_id].add_scan(
                    scan_id, scan_source_id, mjd1, data
                )

        return None
