from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
from pathlib import Path
import datetime
from astropy import time
import numpy as np
from scipy import interpolate
from .. import astro, io, utils
from ..logger import log
from ..constants import CLIGHT
from ..station.core import Station
from ..experiment.observation import Observation
from ..experiment.baseline import Baseline
from ..delays import AntennaDelays, Geometric, Ionospheric, Tropospheric
from ..source import NearFieldSource

if TYPE_CHECKING:
    from ..experiment import Experiment
    from ..displacements.core import Displacement
    from ..types import Band


DISPLACEMENT_CORRECTIONS: tuple[str, ...] = (
    "SolidTide",
    "OceanLoading",
    "PoleTide",
)
"""Displacement models whose contributions are reported as corrections"""

RAMPING_TIME_SCALES: dict[str, str] = {"one-way": "tdb", "three-way": "utc"}
"""Time scale of the epochs in the frequency ramping tables"""


@dataclass
class UplinkLeg:
    """Upleg of a three-way link for a single uplink station

    :param indices: Indices of the reception epochs served by this station
    :param observation: Observation of the target from the uplink station, at the approximate transmission epochs, padded with two extra epochs at each end.
    """

    indices: np.ndarray
    observation: Observation


@dataclass
class SkyFrequency:
    """Sky frequency at a receiving station and associated link geometry

    :param values: Sky frequency at the receiving station [Hz]
    :param rx_observation: Observation of the target from the receiving station
    :param uplink_legs: Upleg data for each uplink station (three-way only)
    :param uplink_stations: Name of the uplink station for each epoch (three-way only)
    :param uplink_frequency: Uplink frequency at the transmitting station for each epoch [Hz] (three-way only)
    """

    values: np.ndarray
    rx_observation: Observation
    uplink_legs: list[UplinkLeg]
    uplink_stations: list[str]
    uplink_frequency: np.ndarray | None = None


class DopplerPredictor:
    """Doppler predictions for the near-field target of an experiment

    Calculates the sky frequency of the target's signal at the receiving stations of the experiment, for a one-way (spacecraft to station) or three-way (station to spacecraft to another station) link. The sky frequency is calculated with topocentric station coordinates, corrected for all the geophysical displacements and antenna delays specified in the configuration file.

    The contributions of the troposphere, the ionosphere, solid Earth tides, ocean loading, pole tide and antenna deformation (thermal deformation and axis offset) to the sky frequency are also reported. For three-way links, each contribution is the sum of the contributions of the receiving and transmitting stations. The tropospheric, ionospheric and antenna contributions use the delay models enabled in the "Delays" section of the configuration file.

    :param exp: Experiment object
    """

    def __init__(self, exp: "Experiment") -> None:

        self.exp = exp
        self.config: dict[str, Any] = exp.setup.doppler

        # Link type
        self.link: str = self.config["link"]
        if self.link not in RAMPING_TIME_SCALES:
            log.error(
                "Failed to initialize Doppler predictions: "
                f"Invalid link type {self.link}. "
                f"Valid options are {list(RAMPING_TIME_SCALES)}"
            )
            exit(1)

        # Time grid
        self.step = float(self.config["step"])
        self.offset = float(self.config.get("offset", 0.0))
        if self.step <= 0.0:
            log.error(
                "Failed to initialize Doppler predictions: "
                f"Invalid step size {self.step}"
            )
            exit(1)

        # Stations for which to calculate predictions [Empty: All]
        self.station_ids: list[str] = self.config.get("stations", [])

        # Near-field target
        target = exp.sources[exp.target["short_name"]]
        assert isinstance(target, NearFieldSource)  # Sanity
        self.target = target

        # Turnaround ratio
        self.turnaround_ratio = 1.0
        if self.link == "three-way":
            uplink_band = self.config["uplink_band"]
            downlink_band = self.config["downlink_band"]
            turnaround_ratio = astro.get_turnaround_ratio(uplink_band, downlink_band)
            if turnaround_ratio is None:
                log.error(
                    "Failed to initialize Doppler predictions: "
                    f"Turnaround ratio not available for {uplink_band}/"
                    f"{downlink_band} uplink/downlink bands"
                )
                exit(1)
            self.turnaround_ratio = turnaround_ratio

        # Frequency ramping data
        self.ramping = self.load_ramping_data()

        # Delay models used for the propagation and antenna contributions. They are
        # used if they are enabled in the "Delays" section of the configuration
        # file (no separate switches for the Doppler predictions)
        self.antenna_model: AntennaDelays | None = None
        self.tropospheric_model: Tropospheric | None = None
        self.ionospheric_model: Ionospheric | None = None
        for model in exp.delay_models:
            if isinstance(model, AntennaDelays):
                self.antenna_model = model
            elif isinstance(model, Tropospheric):
                self.tropospheric_model = model
            elif isinstance(model, Ionospheric):
                self.ionospheric_model = model

        # Geometric model (for the geocentric sky frequency)
        self.geometric_model: Geometric | None = None
        for model in exp.delay_models:
            if isinstance(model, Geometric):
                self.geometric_model = model

        # Reference frequency of the Doppler phase [Hz] (None: no phase)
        self.phase_reference_frequency: float | None = self.config.get(
            "phase_reference_frequency"
        )
        if (
            self.phase_reference_frequency is not None
            and self.geometric_model is None
        ):
            log.error(
                "Failed to initialize Doppler predictions: The Doppler phase "
                "requires the Geometric delay"
            )
            exit(1)

        # Geocentric sky frequency per receiving station:
        # station ID -> (geocentric reception epochs, frequency [Hz])
        self.geocentric_frequencies: dict[str, tuple[time.Time, np.ndarray]] = {}

        # Massive bodies for relativistic corrections
        self.bodies: list[str] = io.internal_parameter("lt_correction_bodies")
        self.external_bodies = [
            body for body in self.bodies if body.lower() != "earth"
        ]

        # Settings for Newton-Raphson algorithm
        self.lt_precision = float(io.internal_parameter("lt_precision"))
        self.lt_max_iterations = int(io.internal_parameter("lt_max_iterations"))

        return None

    def load_ramping_data(self) -> dict[str, Any]:
        """Load frequency ramping data for the link

        For a three-way link, the ramping data contains the uplink frequency at the transmitting station on Earth, and its epochs are in UTC. For a one-way link, the ramping data contains the downlink frequency at the spacecraft, and its epochs are in TDB. The data is loaded with a margin of one day before the start of the experiment, to cover the light-time of the signal.

        :return: Dictionary with ramping data [See io.load_ramping_data]
        """

        ramping_file = io.get_path_to_ramping_data_file(
            self.target.name, self.link  # type: ignore
        )
        margin = time.TimeDelta(1.0, format="jd")
        ramping_data = io.load_ramping_data(
            ramping_file,
            self.link,  # type: ignore
            (self.exp.initial_epoch - margin, self.exp.final_epoch),
            RAMPING_TIME_SCALES[self.link],
        )
        if ramping_data is None:
            log.error(
                "Failed to initialize Doppler predictions: "
                f"No {self.link} ramping data for {self.target.name} "
                f"in {ramping_file}"
            )
            exit(1)

        return ramping_data

    def run(self) -> None:
        """Calculate and save Doppler predictions for all stations

        Must be called with the SPICE kernels of the experiment loaded.
        """

        log.info(f"Calculating {self.link} Doppler predictions")

        # Gravitational parameters of massive bodies
        self.bodies_gm = np.array(
            [astro.get_body_gravitational_parameter(b) for b in self.bodies]
        )
        self.external_bodies_gm = np.array(
            [astro.get_body_gravitational_parameter(b) for b in self.external_bodies]
        )

        # Stations of the experiment
        station_catalog = self.exp.vex.load_station_ids_and_names(
            self.exp.setup.general["ignore_stations"]
        )
        station_ids = self.station_ids if self.station_ids else station_catalog
        for station_id in station_ids:

            if station_id not in station_catalog:
                log.error(
                    "Failed to calculate Doppler predictions: "
                    f"Station {station_id} not found in VEX file"
                )
                exit(1)

            # Reception epochs and observation band for this station
            epochs, band = self.reception_epochs(station_id)
            if len(epochs) == 0:
                log.warning(
                    f"Skipping Doppler predictions for {station_id}: "
                    "No scans of the target"
                )
                continue

            self.process_station(
                station_catalog[station_id], station_id, epochs, band
            )

        return None

    def reception_epochs(
        self, station_id: str
    ) -> tuple[list[datetime.datetime], "Band | None"]:
        """Reception epochs at a station

        Discretizes all the scans of the target in which the station participates, using the step size and offset from the configuration file. For each scan, the epochs go from the start of the scan for the station plus the offset, to the end of the scan plus the offset.

        :param station_id: Two-letter code of the station
        :return epochs: Sorted list of UTC reception epochs
        :return band: Observation band of the station in the first scan
        """

        vex = self.exp.vex
        epochs: list[datetime.datetime] = []
        band: "Band | None" = None
        for scan_id in sorted(vex.experiment_scans_ids):

            scan = vex.load_single_scan_data(
                scan_id, self.exp.target["short_name"]
            )
            if scan.source_type != "target":
                continue
            if station_id not in scan.offsets_per_station:
                continue

            if band is None:
                band = self.exp.modes[scan.observation_mode].get_station_band(
                    station_id
                )

            initial_offset, final_offset = scan.offsets_per_station[station_id]
            start = scan.initial_epoch + datetime.timedelta(
                seconds=initial_offset + self.offset
            )
            n_epochs = int(np.floor((final_offset - initial_offset) / self.step + 1e-9)) + 1
            epochs += [
                start + datetime.timedelta(seconds=idx * self.step)
                for idx in range(n_epochs)
            ]

        return sorted(set(epochs)), band

    def process_station(
        self,
        station_name: str,
        station_id: str,
        epochs: list[datetime.datetime],
        band: "Band | None",
    ) -> None:
        """Calculate and save Doppler predictions for a receiving station

        The contribution of each displacement model to the sky frequency is the difference between the sky frequency calculated with all the models, and the one calculated without that model. The contributions of the tropospheric, ionospheric and antenna delays are calculated from their rates of change at the receiving (and transmitting) stations (see `delay_contribution`). All of them are included in the sky frequency.

        :param station_name: Name of the receiving station
        :param station_id: Two-letter code of the receiving station
        :param epochs: UTC reception epochs
        :param band: Observation band of the receiving station
        """

        log.info(f"Calculating Doppler predictions for {station_name}")

        # Sky frequency with all displacement models
        models = self.exp.displacement_models
        nominal = self.sky_frequency(station_name, station_id, epochs, band, models)

        # Contribution of displacement models
        corrections: list[np.ndarray] = []
        for name in DISPLACEMENT_CORRECTIONS:

            if name not in [model.name for model in models]:
                corrections.append(np.zeros_like(nominal.values))
                continue

            log.info(f"Calculating contribution of {name} for {station_name}")
            reduced_models = [model for model in models if model.name != name]
            reduced = self.sky_frequency(
                station_name, station_id, epochs, band, reduced_models
            )
            corrections.append(nominal.values - reduced.values)

        # Source coordinates at the receiving and uplink stations
        nominal.rx_observation.update_with_source_coordinates()
        for leg in nominal.uplink_legs:
            leg.observation.update_with_source_coordinates()

        # Contribution of tropospheric, ionospheric and antenna delays
        tropospheric_correction = self.delay_contribution(
            nominal, self.tropospheric_model
        )
        ionospheric_correction = self.delay_contribution(
            nominal, self.ionospheric_model, dispersive=True
        )
        antenna_correction = self.delay_contribution(nominal, self.antenna_model)

        # Order of the output columns: troposphere, ionosphere, solid tide,
        # ocean loading, pole tide, antenna
        corrections = [tropospheric_correction, ionospheric_correction] + corrections
        corrections.append(antenna_correction)

        # Geocentric sky frequency (without antenna delays)
        if self.geometric_model is not None:
            self.geocentric_frequencies[station_id] = self.geocentric_frequency(
                nominal
            )

        # Sky frequency including propagation and antenna delays
        sky_frequency = (
            nominal.values
            + tropospheric_correction
            + ionospheric_correction
            + antenna_correction
        )

        self.save_output(
            station_name, station_id, nominal, sky_frequency, corrections
        )

        return None

    def initialize_observation(
        self,
        station: Station,
        epochs: list[datetime.datetime],
        band: "Band | None",
        displacement_models: list["Displacement"],
    ) -> Observation:
        """Observation of the target with geophysically-corrected station

        Creates an observation of the target from a station at a set of epochs, and updates the station with the selected geophysical displacements. The station gets interpolators for its position and velocity, which are valid between the first and last epochs.

        :param station: Station object (not shared with the experiment)
        :param epochs: Sorted list of UTC epochs
        :param band: Observation band
        :param displacement_models: Displacement models to apply
        :return: Observation object
        """

        observation = Observation(station, self.target, band, epochs)  # type: ignore
        baseline = Baseline(self.exp.phase_center, station, [observation])
        baseline.update_station_with_geophysical_displacements(
            displacement_models, self.exp.eops
        )

        return observation

    def sky_frequency(
        self,
        station_name: str,
        station_id: str,
        epochs: list[datetime.datetime],
        band: "Band | None",
        displacement_models: list["Displacement"],
    ) -> SkyFrequency:
        """Sky frequency at a receiving station

        For a one-way link, the sky frequency is the product of the frequency transmitted by the spacecraft (from the one-way ramping table) and the frequency ratio of the downleg. For a three-way link, it is the product of the uplink frequency at the transmitting station (from the three-way ramping table), the turnaround ratio, and the frequency ratios of the upleg and the downleg.

        :param station_name: Name of the receiving station
        :param station_id: Two-letter code of the receiving station
        :param epochs: Sorted list of UTC reception epochs
        :param band: Observation band of the receiving station
        :param displacement_models: Displacement models to apply to stations
        :return: Sky frequency and link geometry
        """

        # Receiving station
        rx_station = Station(
            station_name, station_id, False, self.exp.clock_parameters[station_id]
        )
        rx_observation = self.initialize_observation(
            rx_station, epochs, band, displacement_models
        )
        rx = rx_observation.tstamps

        # Downleg: Spacecraft to receiving station
        #######################################################################

        # Transmission epochs at spacecraft
        tx_sc = self.target.tx_from_rx(rx, rx_station)
        tx_sc = time.Time(tx_sc.tdb.jd1, tx_sc.tdb.jd2, format="jd", scale="tdb")

        # BCRF states of receiver, spacecraft and massive bodies
        et_rx = utils.get_ephemeris_time_from_epoch(rx)
        et_sc = utils.get_ephemeris_time_from_epoch(tx_sc)
        x_rx, v_rx = self.station_bcrf_state(rx_station, rx)
        s_sc = astro.get_icrf_state_vector(self.target.spice_id, et_sc)
        x_sc, v_sc = s_sc[:, :3], s_sc[:, 3:]
        x_bodies_rx, v_bodies_rx = self.bodies_bcrf_state(et_rx)
        x_bodies_sc, v_bodies_sc = self.bodies_bcrf_state(et_sc)

        ratio_down = astro.calculate_frequency_ratio(
            self.bodies_gm,
            x_sc,
            v_sc,
            x_bodies_sc,
            v_bodies_sc,
            x_rx,
            v_rx,
            x_bodies_rx,
            v_bodies_rx,
        )

        if self.link == "one-way":
            f_sc, _ = self.ramped_frequency(tx_sc)
            return SkyFrequency(f_sc * ratio_down, rx_observation, [], [])

        # Upleg: Transmitting station to spacecraft
        #######################################################################

        # Approximate transmission epochs using geocentric light-time
        x_earth_sc = astro.get_icrf_position_vector("earth", et_sc)
        lt_approx = np.linalg.norm(x_sc - x_earth_sc, axis=-1) / CLIGHT
        tx_up_approx = tx_sc - time.TimeDelta(lt_approx, format="sec", scale="tdb")

        # Uplink station for each epoch
        _, uplink_stations = self.ramped_frequency(tx_up_approx)

        ratio_up = np.zeros_like(ratio_down)
        f_up = np.zeros_like(ratio_down)
        uplink_legs: list[UplinkLeg] = []
        for uplink_station_name in sorted(set(uplink_stations)):

            indices = np.where(np.array(uplink_stations) == uplink_station_name)[0]

            # Uplink station with interpolators covering transmission epochs
            tx_station = Station(uplink_station_name, uplink_station_name, False, None)
            tx_guess = self.station_epochs(tx_station, tx_up_approx[indices])
            padding = [
                datetime.timedelta(seconds=k * self.step) for k in (-2, -1, 1, 2)
            ]
            guess_dt = list(tx_guess.utc.to_datetime())  # type: ignore
            padded_epochs = (
                [guess_dt[0] + dt for dt in padding[:2]]
                + guess_dt
                + [guess_dt[-1] + dt for dt in padding[2:]]
            )
            tx_observation = self.initialize_observation(
                tx_station, padded_epochs, band, displacement_models
            )
            uplink_legs.append(UplinkLeg(indices, tx_observation))

            # Transmission epochs at uplink station
            tx_up = self.uplink_transmission_epochs(
                tx_station,
                tx_sc[indices],
                x_sc[indices],
                x_bodies_sc[:, indices],
                tx_up_approx[indices],
            )

            # BCRF states of transmitter and massive bodies
            x_tx, v_tx = self.station_bcrf_state(tx_station, tx_up)
            et_up = utils.get_ephemeris_time_from_epoch(tx_up)
            x_bodies_up, v_bodies_up = self.bodies_bcrf_state(et_up)

            ratio_up[indices] = astro.calculate_frequency_ratio(
                self.bodies_gm,
                x_tx,
                v_tx,
                x_bodies_up,
                v_bodies_up,
                x_sc[indices],
                v_sc[indices],
                x_bodies_sc[:, indices],
                v_bodies_sc[:, indices],
            )

            # Uplink frequency at transmission epochs
            f_up[indices], _stations = self.ramped_frequency(tx_up)
            if any(name != uplink_station_name for name in _stations):
                log.error(
                    "Failed to calculate three-way Doppler: Uplink station "
                    "changes within the light-time correction"
                )
                exit(1)

        values = f_up * self.turnaround_ratio * ratio_up * ratio_down

        return SkyFrequency(
            values, rx_observation, uplink_legs, uplink_stations, f_up
        )

    def uplink_transmission_epochs(
        self,
        tx_station: Station,
        rx_sc: "time.Time",
        x_sc_bcrf: np.ndarray,
        x_bodies_bcrf_sc: np.ndarray,
        tx_guess: "time.Time",
    ) -> "time.Time":
        """Transmission epochs at an uplink station

        Given the reception epochs at the spacecraft, solves the light-time equation of the upleg with the Newton-Raphson algorithm, as formulated in Duev (2012), including post-Newtonian relativistic corrections.

        :param tx_station: Uplink station, with position interpolators covering the transmission epochs
        :param rx_sc: Reception epochs at the spacecraft (TDB)
        :param x_sc_bcrf: BCRF position of the spacecraft at reception (N, 3)
        :param x_bodies_bcrf_sc: BCRF position of the massive bodies at reception (M, N, 3)
        :param tx_guess: Initial guess for the transmission epochs (TDB)
        :return: Transmission epochs at the uplink station (TDB, with location)
        """

        # Initialize light-time
        lt_0: np.ndarray = (rx_sc.tdb - tx_guess.tdb).to("s").value  # type: ignore
        lt_i: np.ndarray = 0.0 * lt_0
        n_i: int = 0

        # Iterative correction of TX epoch
        # Function: F(LT) = LT - R_20/c - RLT_20
        # Derivative: dF/dLT = 1 - (R_20_vec * dR_2_vec/dTX) / (R_20 * c)
        while np.any(np.abs(lt_0 - lt_i) > self.lt_precision) and (
            n_i < self.lt_max_iterations
        ):

            lt_i = lt_0
            tx = self.station_epochs(
                tx_station,
                rx_sc.tdb - time.TimeDelta(lt_i, format="sec", scale="tdb"),
            )

            # BCRF state of transmitter and position of massive bodies at TX
            x_tx, v_tx = self.station_bcrf_state(tx_station, tx)
            et_tx = utils.get_ephemeris_time_from_epoch(tx)
            x_bodies_tx, _ = self.bodies_bcrf_state(et_tx)

            # Relativistic correction to light-time
            rlt = astro.post_newtonian_near_field_effect(
                self.bodies_gm,
                x_sc_bcrf,
                x_tx,
                x_bodies_bcrf_sc,
                x_bodies_tx,
                consider_bending=True,
            )

            # Evaluate function and derivative for Newton-Raphson
            r20 = x_sc_bcrf - x_tx
            r20_mag = np.linalg.norm(r20, axis=-1)
            f = lt_i - r20_mag / CLIGHT - rlt
            dfdlt = 1.0 - np.sum((r20 / r20_mag[:, None]) * v_tx, axis=-1) / CLIGHT

            # Update light-time
            lt_0 = lt_i - f / dfdlt
            n_i += 1

        return self.station_epochs(
            tx_station, rx_sc.tdb - time.TimeDelta(lt_0, format="sec", scale="tdb")
        )

    @staticmethod
    def station_epochs(station: Station, epochs: "time.Time") -> "time.Time":
        """Epochs in TDB with the location of a station

        The location is required for the conversion between TDB and UTC at the station.

        :param station: Station object
        :param epochs: Epochs in any time scale
        :return: Epochs in TDB, with the location of the station
        """

        tdb = epochs.tdb  # type: ignore
        return time.Time(
            tdb.jd1,
            tdb.jd2,
            format="jd",
            scale="tdb",
            location=station.tectonic_corrected_location(tdb),
        )

    def station_bcrf_state(
        self, station: Station, epochs: "time.Time"
    ) -> tuple[np.ndarray, np.ndarray]:
        """BCRF position and velocity of a station

        :param station: Station with geophysical corrections
        :param epochs: Epochs with the location of the station
        :return: BCRF position and velocity of the station as (N, 3) arrays
        """

        utc = epochs.utc  # type: ignore
        x_gcrf = station.location(utc, frame="icrf")
        v_gcrf = station.velocity(utc, frame="icrf")

        # BCRF state of the Earth and position of external bodies
        et = utils.get_ephemeris_time_from_epoch(epochs)
        s_earth = astro.get_icrf_state_vector("earth", et)
        x_external_bodies = np.array(
            [astro.get_icrf_position_vector(body, et) for body in self.external_bodies]
        )

        # Newtonian potential of external bodies at geocenter
        potential = astro.calculate_newtonian_potential_from_bcrf_positions(
            massive_bodies_gm=self.external_bodies_gm,
            x_target_bcrf=s_earth[:, :3],
            x_bodies_bcrf=x_external_bodies,
        )

        x_bcrf = astro.transform_position_from_gcrf_to_bcrf(
            x_gcrf, s_earth[:, :3], s_earth[:, 3:], potential
        )
        v_bcrf = astro.transform_velocity_from_gcrf_to_bcrf(
            v_gcrf, s_earth[:, 3:], potential
        )

        return x_bcrf, v_bcrf

    def bodies_bcrf_state(self, et: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BCRF position and velocity of the massive bodies

        :param et: Ephemeris times (N,)
        :return: BCRF position and velocity of the bodies as (M, N, 3) arrays
        """

        states = np.array([astro.get_icrf_state_vector(body, et) for body in self.bodies])
        return states[:, :, :3], states[:, :, 3:]

    def ramped_frequency(
        self, epochs: "time.Time"
    ) -> tuple[np.ndarray, list[str]]:
        """Frequency from ramping table

        For each epoch, finds the ramping interval such that start <= epoch < end, and calculates the frequency as f_0 + df * (epoch - start). If several intervals contain the epoch, the one that starts last is used.

        :param epochs: Epochs at which to evaluate the frequency
        :return frequency: Ramped frequency [Hz]
        :return stations: Name of the uplink station for each epoch (three-way only)
        """

        t0: time.Time = self.ramping["t0"]
        t1: time.Time = self.ramping["t1"]

        # Time since start and until end of each interval [N, K]
        _epochs = epochs.reshape(-1, 1)  # type: ignore
        dt_start = (_epochs - t0.reshape(1, -1)).to("s").value  # type: ignore
        dt_end = (t1.reshape(1, -1) - _epochs).to("s").value  # type: ignore
        in_interval = (dt_start >= 0.0) & (dt_end > 0.0)

        # Ensure all epochs are covered
        not_covered = ~np.any(in_interval, axis=1)
        if np.any(not_covered):
            log.error(
                "Failed to calculate Doppler predictions: No ramping data "
                f"for {self.target.name} at {epochs[not_covered][0].isot}"  # type: ignore
            )
            exit(1)

        # Index of last interval containing each epoch
        n_intervals = in_interval.shape[1]
        rows = n_intervals - 1 - np.argmax(in_interval[:, ::-1], axis=1)
        cols = np.arange(len(rows))

        frequency = self.ramping["f0"][rows] + self.ramping["df"][rows] * dt_start[cols, rows]
        stations = (
            [self.ramping["uplink"][row] for row in rows]
            if "uplink" in self.ramping
            else []
        )

        return frequency, stations

    def delay_contribution(
        self,
        sky_frequency: SkyFrequency,
        model: Any,
        dispersive: bool = False,
    ) -> np.ndarray:
        """Contribution of a station delay to the sky frequency

        A delay tau(t) added to the path of the signal at a station changes the received frequency by -f * dtau/dt. The contribution is calculated for the receiving station and, for a three-way link, for the uplink station(s), and summed:

            delta_f = sign * f * (dtau_rx/dt + dtau_tx/dt)

        where f is the (geometric) sky frequency. For non-dispersive delays (troposphere, antenna) sign = -1. For the ionosphere (dispersive), the model gives the group delay, while the Doppler is measured on the carrier phase, which the ionosphere advances by the same amount: sign = +1. The ionospheric delay is evaluated at the downlink frequency (sky frequency) for the receiving station, and at the uplink frequency for the uplink station. NOTE: Dima's code (pypride) uses -1 for the ionosphere too.

        The uplink station's delay is evaluated at the approximate transmission epochs (observation of the upleg, padded with two epochs at each end), which differ from the final ones by a few ms.

        :param sky_frequency: Sky frequency and link geometry (source coordinates of the observations must be set)
        :param model: Delay model (AntennaDelays, Tropospheric or Ionospheric), or None
        :param dispersive: True for the ionosphere
        :return: Contribution to the sky frequency [Hz]
        """

        values = sky_frequency.values
        if model is None:
            return np.zeros_like(values)

        # Receiving station: downlink frequency
        rx_observation = sky_frequency.rx_observation
        if dispersive:
            rx_delay = model.calculate(rx_observation, frequency=values)
        else:
            rx_delay = model.calculate(rx_observation)
        rate = self.delay_rate(rx_observation.tstamps, rx_delay)

        # Uplink stations (excluding padding): uplink frequency
        for leg in sky_frequency.uplink_legs:
            tx_observation = leg.observation
            if dispersive:
                assert sky_frequency.uplink_frequency is not None  # Sanity
                f_leg = sky_frequency.uplink_frequency[leg.indices]
                f_padded = np.r_[f_leg[:1], f_leg[:1], f_leg, f_leg[-1:], f_leg[-1:]]
                tx_delay = model.calculate(tx_observation, frequency=f_padded)
            else:
                tx_delay = model.calculate(tx_observation)
            rate[leg.indices] += self.delay_rate(tx_observation.tstamps, tx_delay)[2:-2]

        sign = 1.0 if dispersive else -1.0
        return sign * values * rate

    def geocentric_frequency(
        self, sky_frequency: SkyFrequency
    ) -> tuple["time.Time", np.ndarray]:
        """Sky frequency at the geocenter

        Converts the topocentric sky frequency at the receiving station (geometric: without tropospheric, ionospheric and antenna delays) to the geocenter with the geometric delay of Alfonso's code (`Geometric`), tau = t_station - t_geocenter. A wavefront that reaches the station at t_s reaches the geocenter at t_g = t_s - tau(t_s), so the frequencies are related by f_g = f_s * dt_s/dt_g = f_s / (1 - dtau/dt_s).

        :param sky_frequency: Topocentric sky frequency and link geometry
        :return epochs: Reception epochs at the geocenter (UTC)
        :return frequency: Geocentric sky frequency [Hz]
        """

        assert self.geometric_model is not None  # Sanity
        observation = sky_frequency.rx_observation
        tau = self.geometric_model.calculate(observation)
        rate = self.delay_rate(observation.tstamps, tau)

        utc = observation.tstamps.utc  # type: ignore
        epochs = time.Time(utc.jd1, utc.jd2, format="jd", scale="utc") - (
            time.TimeDelta(tau, format="sec")
        )

        return epochs, sky_frequency.values / (1.0 - rate)

    def doppler_phase(
        self, station_id: str, epochs: "time.Time"
    ) -> np.ndarray | None:
        """Accumulated Doppler phase for a scan of the target

        phase(t) = 2 pi * integral from epochs[0] to t of (f_0 - f_gc) dt, where f_gc is the geocentric sky frequency and f_0 the reference frequency of the configuration file (`phase_reference_frequency`). The sign follows Dima's code (pypride, `bindel.phaseCor` and `bindel.dump`). The integral is calculated analytically from a cubic spline of f_0 - f_gc through the geocentric sky frequencies around the scan; the epochs are treated as reception epochs at the geocenter.

        :param station_id: Two-letter code of the receiving station
        :param epochs: UTC epochs of a single scan (sorted)
        :return: Phase [rad] at the epochs, or None if not available (no reference frequency or no Doppler predictions for the station)
        """

        if self.phase_reference_frequency is None:
            return None
        if station_id not in self.geocentric_frequencies:
            return None

        f_epochs, f_gc = self.geocentric_frequencies[station_id]

        # Geocentric frequencies around the scan
        t = (f_epochs - epochs[0]).to("s").value  # type: ignore
        t_end = (epochs[-1] - epochs[0]).to("s").value  # type: ignore
        margin = 2.0 * self.step
        mask = (t >= -margin) & (t <= t_end + margin)
        if np.sum(mask) < 4:
            log.error(
                f"Failed to calculate Doppler phase for {station_id} at "
                f"{epochs[0].isot}: Not enough Doppler predictions in the scan"
            )
            exit(1)

        # Integrate f_0 - f_gc with a cubic spline
        spline = interpolate.CubicSpline(
            t[mask], self.phase_reference_frequency - f_gc[mask]
        )
        integral = spline.antiderivative()
        t_scan = (epochs - epochs[0]).to("s").value  # type: ignore

        return 2.0 * np.pi * (integral(t_scan) - integral(0.0))

    def delay_rate(self, epochs: "time.Time", delay: np.ndarray) -> np.ndarray:
        """Rate of change of a delay

        Calculated with finite differences, separately for each block of consecutive epochs (separated by less than 1.5 times the step size).

        :param epochs: Sorted epochs (N,)
        :param delay: Delay at the epochs [s] (N,)
        :return: Rate of change of the delay [s/s] (N,)
        """

        t = (epochs - epochs[0]).to("s").value  # type: ignore
        breaks = np.where(np.diff(t) > 1.5 * self.step)[0] + 1
        rate = np.zeros_like(delay)
        for block in np.split(np.arange(len(t)), breaks):
            if len(block) < 2:
                log.warning("Delay rate set to zero for isolated epoch")
                continue
            rate[block] = np.gradient(
                delay[block], t[block], edge_order=2 if len(block) > 2 else 1
            )

        return rate

    def save_output(
        self,
        station_name: str,
        station_id: str,
        nominal: SkyFrequency,
        sky_frequency: np.ndarray,
        corrections: list[np.ndarray],
    ) -> None:
        """Write Doppler predictions for a receiving station to a text file"""

        outdir = Path(self.exp.setup.general["output_directory"]).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        output_file = outdir / f"{self.exp.name}_{station_id.title()}_doppler.txt"

        header = [
            f"Doppler predictions for {self.target.name.upper()} in experiment "
            f"{self.exp.name}",
            f"Receiving station: {station_name} ({station_id.title()})",
            f"Link: {self.link}",
        ]
        if self.link == "three-way":
            header += [
                f"Uplink band: {self.config['uplink_band']}",
                f"Downlink band: {self.config['downlink_band']}",
                f"Turnaround ratio: {self.turnaround_ratio:.16f}",
                f"Uplink stations: {', '.join(sorted(set(nominal.uplink_stations)))}",
            ]
        header += [
            f"Step: {self.step} s, offset: {self.offset} s",
            "Columns:",
            "  1: UTC epoch of reception (ISO 8601)",
            "  2: Modified Julian Date (integer part)",
            "  3: UTC seconds past midnight",
            "  4: Ephemeris time (TDB seconds past J2000)",
            "  5: Topocentric sky frequency [Hz] (includes columns 6-11)",
            "  6: Contribution of the troposphere [Hz]",
            "  7: Contribution of the ionosphere [Hz] (carrier phase advance)",
            "  8: Contribution of solid Earth tides [Hz]",
            "  9: Contribution of ocean loading [Hz]",
            " 10: Contribution of pole tide [Hz]",
            " 11: Contribution of antenna deformation (thermal deformation + "
            "axis offset) [Hz]",
            "Contributions are summed over receiving and transmitting stations",
            "Troposphere, ionosphere and antenna: zero if the delay model is off in "
            "the 'Delays' section",
        ]

        io.write_doppler_file(
            output_file,
            header,
            nominal.rx_observation.tstamps,
            sky_frequency,
            corrections,
        )

        return None
