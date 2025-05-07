from typing import TYPE_CHECKING, Generator, Any
from pathlib import Path
from contextlib import contextmanager
import spiceypy as spice
from ..io import (
    Setup,
    Vex,
    VexContent,
    load_catalog,
    VEX_DATE_FORMAT,
    get_target_information,
    DelFile,
)
from ..logger import log
from astropy import time
import datetime
from ..types import ObservationMode, Band
from ..coordinates import EOP
from ..displacements import DISPLACEMENT_MODELS
from ..delays import DELAY_MODELS
from .source import Source, NearFieldSource, FarFieldSource
from .station import Station
from .baseline import Baseline
from .observation import Observation
import math
import numpy as np

if TYPE_CHECKING:
    from ..displacements.core import Displacement
    from ..delays.core import Delay


class Experiment:
    """VLBI experiment"""

    def __init__(self, setup: str | Path) -> None:

        # Load setup and VEX file
        self.setup, self.vex = self.__load_setup_and_vex(setup)

        # Read basic metadata from VEX file
        self.name, self.initial_epoch, self.final_epoch = (
            self.__get_experiment_metadata()
        )
        log.info(f"Initializing {self.name} experiment")

        # Observation modes
        # TODO: How is this information used?
        self.modes: dict[str, "ObservationMode"] = {
            mode: ObservationMode(
                mode, self.vex["MODE"][mode].getall("FREQ"), self.vex["FREQ"]
            )
            for mode in self.vex["MODE"]
        }

        # Clock offsets
        # TODO: How is this information used?
        self.clock_offsets = self.load_clock_offsets()

        # EOPs: For transformations between ITRF and ICRF
        self.eops = EOP.from_experiment(self)

        # Load target information
        self.target = get_target_information(self.setup.general["target"])

        # Load sources
        self.sources = self.load_sources()

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
        self.baselines = self.initialize_baselines()

        # Initialize delay and displacement models
        self.requires_spice = False
        self.displacement_models = self.initialize_displacement_models()
        self.delay_models = self.initialize_delay_models()

        return None

    def __load_setup_and_vex(
        self, _setup: str | Path
    ) -> tuple[Setup, "VexContent"]:
        """Load information from configuration and VEX files

        Information from the configuration file is loaded into a `Setup` object,
        and the VEX file is parsed for easy access to its contents.

        :param setup: Path to the configuration file
        :return Setup: Setup object containing the configuration
        :return VexContent: VEX object containing the parsed VEX file
        """

        # Load setup from configuration file
        _setup_path = Path(_setup).absolute()
        if not _setup_path.is_file():
            log.error(
                f"Failed to initialize {self.name} experiment: "
                f"Configuration file {_setup_path} not found"
            )
            exit(1)
        setup = Setup(str(_setup_path))

        # Parse VEX file
        _vex_path = _setup_path.parent / setup.general["vex"]
        if not _vex_path.is_file():
            log.error(
                f"Failed to initialize {self.name} experiment: "
                f"VEX file {_vex_path} not found"
            )
            exit(1)
        vex = Vex(str(_vex_path))

        return setup, vex

    def __get_experiment_metadata(self) -> tuple[str, "time.Time", "time.Time"]:
        """Retrieve experiment metadata from VEX

        Reads the experiment name, and the initial and final epochs from the VEX file.

        :return name: Name of the experiment
        :return initial_epoch: Initial epoch of the experiment (UTC)
        :return final_epoch: Final epoch of the experiment (UTC)
        """

        # Experiment name
        name = self.vex["GLOBAL"]["EXPER"]

        # Initial and final epochs
        initial_epoch = time.Time.strptime(
            self.vex["EXPER"][name]["exper_nominal_start"],
            self.setup.internal["vex_date_format"],
            scale="utc",
        )
        final_epoch = time.Time.strptime(
            self.vex["EXPER"][name]["exper_nominal_stop"],
            self.setup.internal["vex_date_format"],
            scale="utc",
        )

        return name, initial_epoch, final_epoch

    def load_sources(self) -> dict[str, "Source"]:
        """Load sources from VEX file

        Parses the SOURCE section of the VEX file and retrieves the name, type and coordinates of all the sources involved in the experiment. NearFieldSource and FarFieldSource objects are initialized for each source based on the 'source_type' read from the VEX, with the program raising an error if this attribute is not set.

        :return: Dictionary with source name as key and Source object as value.
        """

        sources: dict[str, "Source"] = {}
        # nearfield_source_ra: list[str] = []
        # nearfield_source_dec: list[str] = []
        # nearfield_source_frame: list[str] = []

        # # Classify sources
        # for name, source_info in self.vex["SOURCE"].items():

        #     # Ensure type information is available in VEX file
        #     if "source_type" not in source_info:
        #         log.error(
        #             f"Failed to generate {name} source: "
        #             "Type information missing from VEX file"
        #         )
        #         exit(1)

        #     match source_info["source_type"]:
        #         case "calibrator":
        #             sources[name] = FarFieldSource.from_experiment(name, self)
        #         case "target":
        #             nearfield_source_ra.append(source_info["ra"])
        #             nearfield_source_dec.append(source_info["dec"])
        #             nearfield_source_frame.append(
        #                 source_info["ref_coord_frame"]
        #             )
        #         case _:
        #             log.error(
        #                 f"Failed to generate {name} source: "
        #                 f"Invalid type {source_info['source_type']}"
        #             )

        # # Load near field source
        # # NOTE: It is assumed that only one spacecraft is observed
        # for name, source_info in _nearfield_source_data.items():

        #     print(source_info)
        #     exit(0)

        for name, source_info in self.vex["SOURCE"].items():

            # Ensure type information is available in VEX file
            if "source_type" not in source_info:
                log.error(
                    f"Failed to generate {name} source: "
                    "Type information missing from VEX file"
                )
                exit(1)

            # Ensure that coordinates are given in the right frame
            if source_info["ref_coord_frame"] != "J2000":
                raise NotImplementedError(
                    f"Failed to generate {name} source: "
                    f"Invalid reference frame {source_info['ref_coord_frame']}"
                )

            # Initialize far field sources
            match source_info["source_type"]:
                case "calibrator":
                    sources[name] = FarFieldSource.from_experiment(self, name)
                case "target":
                    pass  # Initialized later
                case _:
                    log.error(
                        f"Failed to generate {name} source: "
                        f"Invalid type {source_info['source_type']}"
                    )
                    exit(1)

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
                displacement_models.append(
                    DISPLACEMENT_MODELS[displacement](self)
                )
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

    def initialize_baselines(self) -> list["Baseline"]:
        """Initialize baselines

        Parses the STATION section of the VEX file to identify all the observatories potentially involved in the experiment. Checks the names of the antennas against a station catalog and, if necessary, modifies them to ensure that they match the ID used in the external data files of the delay and displacement models. Each station is then used to initialize a Baseline object using the 'phase_center' attribute of the experiment class.

        The program then parses the SCHED section of the VEX file to get the stations, observation mode, source and time window of each scan. The time windows of the scans are discretized according to the step size, and the limits for the number of time stamps specified in the internal configuration file; and then groupped per station and source. This information is then used to initialize Observation objects, which are saved in the 'observations' attribute of each baseline.

        :return: List of baseline objects equipped with observations.
        """

        log.info("Initializing baselines")

        # Load experiment stations
        _experiment_stations = {
            sta: self.vex["STATION"][sta]["ANTENNA"]
            for sta in self.vex["STATION"]
            if sta not in self.setup.general["ignore_stations"]
        }

        # Load station catalog
        _station_catalog = load_catalog("station_names.yaml")

        # Ensure that all the stations have valid names
        for id, exp_sta in _experiment_stations.items():
            for sta_name, sta_alternatives in _station_catalog.items():
                if exp_sta in sta_alternatives:
                    _experiment_stations[id] = sta_name
                    break

        # Define baselines
        _ids = reversed(list(_experiment_stations.keys()))
        _names = reversed(list(_experiment_stations.values()))
        _baselines: dict[str, "Baseline"] = {
            k: Baseline(self.phase_center, Station.from_experiment(v, k, self))
            for k, v in reversed(dict(zip(_ids, _names)).items())
        }

        # Load data about observations
        observation_bands: dict[str, dict[str, "Band"]] = {}
        observation_tstamps: dict[str, dict[str, list[datetime.datetime]]] = {}
        for scan_id, scan_data in self.vex["SCHED"].items():

            # Metadata
            scan_stations = scan_data.getall("station")
            base_start = datetime.datetime.strptime(
                scan_data["start"], self.setup.internal["vex_date_format"]
            )
            mode = self.modes[scan_data["mode"]]
            if len(scan_data.getall("source")) > 1:
                raise NotImplementedError(
                    f"Scan {scan_id} has multiple sources"
                )
            source = scan_data.getall("source")[0]

            # Ensure unique name for all near field sources
            source_type = self.vex["SOURCE"][source]["source_type"]
            if source_type == "target":
                source = self.target["short_name"]

            for station_data in scan_stations:

                # Retrieve station code and offsets of observation window
                _station_code, _dt_start, _dt_end = station_data[:3]
                if _station_code in self.setup.general["ignore_stations"]:
                    continue

                # Identify baseline containing the station
                if _station_code not in _baselines:
                    log.error(
                        f"Failed to initialize baselines for {self.name} "
                        "experiment: No baseline includes "
                        f"station {_station_code}"
                    )
                    exit(1)

                # Time window of the scan
                scan_start = base_start + datetime.timedelta(
                    seconds=int(_dt_start.split()[0])
                )
                scan_end = base_start + datetime.timedelta(
                    seconds=int(_dt_end.split()[0])
                )
                scan_duration = (scan_end - scan_start).total_seconds()

                # Define step size for scan discretization
                default_step: float = self.setup.internal["default_scan_step"]
                min_nobs = self.setup.internal["min_obs_per_scan"]
                min_step = self.setup.internal["min_scan_step"]
                _scan_step = scan_duration / min_nobs
                if _scan_step > default_step:
                    nobs = math.ceil(scan_duration / default_step)
                    scan_step = scan_duration / nobs
                elif min_step <= _scan_step <= default_step:
                    nobs = math.ceil(scan_duration / _scan_step)
                    scan_step = scan_duration / nobs
                else:
                    nobs = math.floor(scan_duration / min_step)
                    scan_step = scan_duration / nobs
                    log.warning(f"Using minimum step size for scan {scan_id}")

                # Discretize scan time window
                scan_step = datetime.timedelta(seconds=scan_step)
                scan_tstamps = [scan_start + i * scan_step for i in range(nobs)]
                scan_tstamps.append(scan_end)

                # Update observation data
                if _station_code not in observation_bands:
                    observation_bands[_station_code] = {}
                    observation_tstamps[_station_code] = {}

                band = mode.get_station_band(_station_code)
                if source not in observation_bands[_station_code]:
                    observation_bands[_station_code][source] = band
                    observation_tstamps[_station_code][source] = scan_tstamps
                else:
                    assert observation_bands[_station_code][source] == band
                    observation_tstamps[_station_code][source] += scan_tstamps

        # Update baselines with observations
        log.info("Updating baselines with observations")
        for baseline_id in observation_bands:

            # Update baselines with observations
            for source_id in observation_bands[baseline_id]:
                _baselines[baseline_id].add_observation(
                    Observation.from_experiment(
                        _baselines[baseline_id],
                        self.sources[source_id],
                        observation_bands[baseline_id][source_id],
                        observation_tstamps[baseline_id][source_id],
                        self,
                    )
                )

        return list(_baselines.values())

    def load_clock_offsets(self):
        """Load clock offset data from VEX"""

        if "CLOCK" not in self.vex:
            return {}

        clock: dict[str, tuple[datetime.datetime, float, float]] = {}
        for station, data in self.vex["CLOCK"].items():
            _offset, _epoch, _rate = data["clock_early"][1:4]
            epoch = datetime.datetime.strptime(_epoch, VEX_DATE_FORMAT)
            offset = float(_offset.split()[0]) * 1e-6
            rate = float(_rate) * 1e-6
            clock[station.title()] = (epoch, offset, rate)
        return clock

    @contextmanager
    def spice_kernels(self) -> Generator:
        """Context manager to load SPICE kernels"""

        try:
            if self.requires_spice:
                log.debug("Loaded SPICE kernels")
                metak = str(
                    self.setup.resources["ephemerides"]
                    / self.setup.general["target"]
                    / "metak.tm"
                )
                spice.furnsh(metak)

            yield None
        finally:
            if self.requires_spice:
                log.debug("Unloaded SPICE kernels")
                spice.kclear()

        return None

    def save_output(self) -> None:

        # Initialize output directory
        outdir = Path(self.setup.general["output_directory"]).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        # Output files and observations
        observations: dict[tuple[str, str], "Observation"] = {}
        output_files: dict[str, "DelFile"] = {}
        for baseline in self.baselines:

            # Station code
            code = baseline.station.id.title()

            # Initialize output file for observation
            output_files[code] = DelFile(outdir / f"{self.name}_{code}.del")
            output_files[code].create_file(code)

            # Add observations to dictionary
            for observation in baseline.observations:
                observations[(code, observation.source.name)] = observation

        # Main loop
        sorted_scans = sorted([s for s in self.vex["SCHED"]])
        for scan_id in sorted_scans:

            # IDs of stations involved in scan
            scan_data = self.vex["SCHED"][scan_id].getall("station")
            scan_stations = [s[0] for s in scan_data]

            # VEX-file and internal ID of source
            _scan_sources = [self.vex["SCHED"][scan_id]["source"]]
            if len(_scan_sources) != 1:
                log.error(f"Scan {scan_id} has multiple sources")
                exit(1)
            scan_source_id = _scan_sources[0]  # ID from VEX file

            _source_data = self.vex["SOURCE"][scan_source_id]
            if _source_data["source_type"] == "target":
                scan_source = self.target["short_name"]  # Internal ID
            else:
                scan_source = scan_source_id  # Internal ID

            # Loop over stations in scan
            for station_id in scan_stations:

                # Get observation for this scan
                assert (station_id, scan_source) in observations  # Sanity
                observation = observations[(station_id, scan_source)]

                # Initial and final epochs for scan
                t0 = time.Time.strptime(
                    self.vex["SCHED"][scan_id]["start"], VEX_DATE_FORMAT
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
                output_files[station_id].add_scan(scan_source_id, mjd1, data)

        return None
