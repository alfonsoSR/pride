from .parser import parse
from dataclasses import dataclass
from pathlib import Path
from ...logger import log
from astropy import time
from ...types import ObservationMode
import datetime
from ..resources import load_catalog

VEX_DATE_FORMAT = "%Yy%jd%Hh%Mm%Ss"


@dataclass
class ScanData:

    source_type: str
    source_name: str
    observation_mode: str
    initial_epoch: datetime.datetime
    offsets_per_station: dict[str, tuple[int, int]]


class Vex:
    """Interface to VEX files."""

    def __init__(self, file: str | Path) -> None:
        """Interface for VEX files

        :param file: Path to the VEX file
        """
        # Convert to Path and ensure it exists
        file = Path(file).absolute()
        if not file.exists():
            log.error(f"Failed to load VEX file: {file} does not exist")
            exit(1)

        # Ensure file has a valid extension
        if file.suffix not in (".vex", ".vix"):
            log.error(
                f"Failed to load VEX file: {file} has unexpected extension"
            )
            exit(1)

        # Load raw content as multi-dict
        log.debug(f"Loading VEX file: {file}")
        self.__content = parse(file.read_text().replace("\r\n", "\n"))

        # Experiment metadata
        self.experiment_name = str(self.__content["GLOBAL"]["EXPER"])
        self.experiment_start = time.Time.strptime(
            self.__content["EXPER"][self.experiment_name][
                "exper_nominal_start"
            ],
            VEX_DATE_FORMAT,
            scale="utc",
        )
        self.experiment_stop = time.Time.strptime(
            self.__content["EXPER"][self.experiment_name]["exper_nominal_stop"],
            VEX_DATE_FORMAT,
            scale="utc",
        )

        # Experiment sources
        self.experiment_sources = self.__content["SOURCE"]

        # Experiment scans
        self.experiment_scans_ids: list[str] = list(
            self.__content["SCHED"].keys()
        )

        return None

    def load_observation_modes(self) -> dict[str, "ObservationMode"]:
        """Load observation modes from the VEX file

        An observation mode is a configuration for the experiment, and it
        specifies the frequency band in which each of the antennas will operate.
        The VEX file can contain multiple observation modes.

        This function reads information about the observation modes from the VEX
        into ObservationMode objects, and returns a dictionary in which each of
        these objects is indexed by the mode ID.

        :return: Dictionary of observation modes
        """

        observation_modes: dict[str, "ObservationMode"] = {}
        experiment_frequency_bands = self.__content["FREQ"]

        for mode_id, mode_content in self.__content["MODE"].items():

            mode_frequency_bands = mode_content.getall("FREQ")
            observation_modes[mode_id] = ObservationMode(
                mode_id,
                mode_frequency_bands,
                experiment_frequency_bands,
            )

        return observation_modes

    def load_clock_parameters(
        self, required: bool = True
    ) -> dict[str, tuple[datetime.datetime, float, float]] | None:
        """Load clock offsets from $CLOCK section

        [From VEX documentation]
        The $CLOCK section contains the necessary clock parameters for proper
        correlation of the data. Normally, this information will be taken from
        the station logs, transcribed manually or, as a last resort, determined
        from a fringe search at the correlator.
        [END]
        The returned dictionary has the ID of a station as key, and a tuple with
        the following items as value:
        - Epoch of origin for the clock model
        - Clock offset with respect to UTC [s]
        - Clock rate [s/s]

        :param required: Fail if $CLOCK section is missing from VEX file
        :return clock_parameters: Dictionary with clock parameters for each
        station, or None if the section is not present and required is False.
        """

        # Check if $CLOCK section is present
        if "CLOCK" not in self.__content:
            if required:
                log.error("$CLOCK section not found in VEX file.")
                exit(1)
            else:
                return None

        # Initialize container for clock parameters
        clock_parameters: dict[str, tuple[datetime.datetime, float, float]] = {}
        for station_id, clock_data in self.__content["CLOCK"].items():

            # Read data and put it in the correct format
            _offset, _epoch, _rate = clock_data["clock_early"][1:4]
            epoch = datetime.datetime.strptime(_epoch, VEX_DATE_FORMAT)
            offset = float(_offset.split()[0]) * 1e-6
            rate = float(_rate) * 1e-6

            # Add entry to dictionary
            clock_parameters[station_id.title()] = (epoch, offset, rate)

        return clock_parameters

    def load_station_ids_and_names(
        self, ignored_stations: list[str] | None = None
    ) -> dict[str, str]:
        """Load information from the $STATION section

        Parses the STATION section of the VEX file to identify all the observatories
        potentially involved in the experiment. Checks the names of the antennas
        against a station catalog, normalizes them to ensure that they match the
        one used in the external data files of the delay and displacement models,
        and ensures that all of the IDs and names are unique.

        :param ignored_stations: List with IDs of stations to ignore
        :return stations: Dictionary with the station IDs as keys, and their
        normalized names as values.
        """

        # Turn ignored stations into a list if None
        if ignored_stations is None:
            ignored_stations = []

        # Initialize dictionary with default station names
        stations_dictionary: dict[str, str] = {
            station_id: self.__content["STATION"][station_id]["ANTENNA"]
            for station_id in self.__content["STATION"]
            if station_id not in ignored_stations
        }

        # Load internal catalog with alternative station names
        alternative_names_catalog = load_catalog("station_names.yaml")

        # Normalize station names
        for station_id, default_name in stations_dictionary.copy().items():
            for (
                normalized_name,
                alternatives,
            ) in alternative_names_catalog.items():
                if default_name in alternatives:
                    print(
                        f"Using {normalized_name} for {station_id} not {default_name}"
                    )
                    stations_dictionary[station_id] = normalized_name
                    break

        # Ensure that there are no duplicates (different IDs for the same name)
        if len(stations_dictionary) != len(set(stations_dictionary.values())):
            log.error("VEX file contains multiple IDs for the same station.")
            exit(1)

        return stations_dictionary

    def load_single_scan_data(self, scan_id: str) -> ScanData:
        """Return data about a single scan

        Returns a ScanData object with the following information:
        - Source type
        - Source name
        - Observation mode
        - Initial epoch for all stations
        - Initial and final time offsets for each station involved

        :param scan_id: ID of the scan to load
        :return scan_data: Data structure with information about the scan
        """

        # Read scan data as mutli-dict
        if scan_id not in self.__content["SCHED"]:
            log.error(f"Scan {scan_id} not found in VEX file.")
            exit(1)
        scan_data = self.__content["SCHED"][scan_id]

        # Get source name
        _sources = scan_data.getall("source")
        if len(_sources) != 1:
            log.error(f"Found multiple sources for scan {scan_id}: {_sources}")
            exit(1)
        source_name = _sources[0]

        # Get source type
        if "source_type" not in self.__content["SOURCE"][source_name]:
            log.error(
                f"Missing `source_type` for source {source_name} in VEX file."
            )
            exit(1)
        source_type = self.__content["SOURCE"][source_name]["source_type"]

        # Get observation mode
        observation_mode = scan_data["mode"]

        # Get reference initial epoch
        initial_epoch: datetime.datetime = datetime.datetime.strptime(
            scan_data["start"], VEX_DATE_FORMAT
        )

        # Get time interval for each station
        offsets_per_station: dict[str, tuple[int, int]] = {}
        for station_data in scan_data.getall("station"):
            station_id, initial_offset, final_offset = station_data[:3]
            offsets_per_station[station_id] = (
                int(initial_offset.split()[0]),
                int(final_offset.split()[0]),
            )

        return ScanData(
            source_type,
            source_name,
            observation_mode,
            initial_epoch,
            offsets_per_station,
        )
