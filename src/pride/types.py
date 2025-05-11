"""Containers and auxiliary data types meant to be used internally"""

from dataclasses import dataclass
from .logger import log
from typing import Literal, Any, Mapping
from enum import Enum


@dataclass
class Channel:
    """Channel

    :param id: Channel ID
    :param sky_freq: Sky frequency [Hz]
    :param net_sideband: Net sideband for downconversion [U/L]
    :param bandwidth: Bandwidth [Hz]
    :param bbc_id: BBC ID
    :param phase_cal_id: Phase calibration ID
    """

    id: str
    sky_freq: float
    net_sideband: Literal["U", "L"]
    bandwidth: float
    bbc_id: str
    phase_cal_id: str

    def __repr__(self) -> str:
        return f"{self.id}: {self.sky_freq} {self.net_sideband} {self.bandwidth} {self.bbc_id} {self.phase_cal_id}"


@dataclass
class Band:
    """Band

    :param name: Band name
    :param stations: List of stations operating in the band
    :param channels: List of channels
    """

    name: str
    stations: list[str]
    channels: list[Channel]

    def __repr__(self) -> str:

        out = f"Band {self.name}\n  Stations: {self.stations}\n  Channels:\n"
        for channel in self.channels:
            out += f"    {channel}\n"
        return out


class ObservationMode:
    """Observation mode

    :param id: Name that identifies the mode
    :param mode_bands: Information about the bands in the mode
    :param experiment_bands: Available bands for current experiment
    """

    def __init__(
        self,
        id: str,
        mode_bands: list[list[str]],
        experiment_bands: Mapping,
    ) -> None:

        band_names = [item[0] for item in mode_bands]
        band_stations = {item[0]: item[1:] for item in mode_bands}
        band_channels = {}

        for band in band_stations:

            band_channels[band] = [
                Channel(
                    id=channel[4],
                    sky_freq=float(channel[1].split()[0]) * 1e6,
                    net_sideband=channel[2],
                    bandwidth=float(channel[3].split()[0]) * 1e6,
                    bbc_id=channel[5],
                    phase_cal_id=channel[6],
                )
                for channel in experiment_bands[band].getall("chan_def")
            ]

        self.id = id
        self.bands: dict[str, Band] = {
            name: Band(name, band_stations[name], band_channels[name])
            for name in band_names
        }

        return None

    def get_station_band(self, station_id: str) -> Band:
        """Get the band in which a station operates

        :param station_id: Two-letter code identifying the station
        """

        for band in self.bands.values():
            if station_id in band.stations:
                return band

        raise ValueError(f"Station {station_id} not found in any band")

    def __eq__(self, other: object) -> bool:
        """Check if two observation modes are equal

        :param other: Other observation mode
        :return: True if the observation modes are equal, False otherwise
        """

        try:
            assert isinstance(other, ObservationMode)
            assert self.id == other.id
            assert self.bands.keys() == other.bands.keys()
            assert all(
                self.bands[band] == other.bands[band] for band in self.bands
            )
        except AssertionError:
            return False

        return True


class SourceType(Enum):
    """Types of observable sources

    Defines the model used to compute geometric delays and the applicability of Doppler corrections
    """

    FarField = 1
    NearField = 2


@dataclass(repr=False)
class Antenna:
    """Container for antenna parameters

    Details in: Nothnagel (2009) https://doi.org/10.1007/s00190-008-0284-z

    Parameter descriptions from: antenna-info.txt (https://ivscc.gsfc.nasa.gov/IVS_AC/IVS-AC_data_information.htm)

    :param ivs_name: IVS station name
    :param focus_type: Focus type of the primary frequency
    :param mount_type: Mounting type
    :param radome: Whether the station has a radome
    :param meas_type: Measurement type (Complete, Incomplete, Rough)
    :param T0: Reference temperature [C]
    :param sin_T: Sine amplitude of annual temperature variations wrt J2000 epoch [C]
    :param cos_T: Cosine amplitude of annual temperature variations wrt J2000 epoch [C]
    :param h0: Reference pressure [hPa]
    :param ant_diam: Antenna diameter [m]
    :param hf: Height of the foundation [m]
    :param df: Depth of the foundation [m]
    :param gamma_hf: Thermal expansion coefficient of the foundation [1/K]
    :param hp: Length of the fixed axis [m]
    :param gamma_hp: Thermal expansion coefficient of the fixed axis [1/K]
    :param AO: Length of the offset between primary and secondary axes [m]
    :param gamma_AO: Thermal expansion coefficient of the offset [1/K]
    :param hv: Distance from the movable axis to the antenna vertex [m]
    :param gamma_hv: Thermal expansion coefficient of the structure connecting the movable axis to the antenna vertex [1/K]
    :param hs: Height of the subreflector/primary focus above the vertex [m]
    :param gamma_hs: Thermal expansion coefficient of the subreflector/primary focus mounting legs [1/K]
    """

    ivs_name: str = NotImplemented
    focus_type: str = NotImplemented
    mount_type: Literal[
        "MO_AZEL", "FO_PRIM", "MO_EQUA", "MO_XYNO", "MO_XYEA", "MO_RICH"
    ] = NotImplemented
    radome: bool = NotImplemented
    meas_type: Literal["ME_COMP", "ME_INCM", "ME_ROUG"] = NotImplemented
    T0: float = NotImplemented
    sin_T: float = NotImplemented
    cos_T: float = NotImplemented
    h0: float = NotImplemented
    ant_diam: float = NotImplemented
    hf: float = NotImplemented
    df: float = NotImplemented
    gamma_hf: float = NotImplemented
    hp: float = NotImplemented
    gamma_hp: float = NotImplemented
    AO: float = NotImplemented
    gamma_AO: float = NotImplemented
    hv: float = NotImplemented
    gamma_hv: float = NotImplemented
    hs: float = NotImplemented
    gamma_hs: float = NotImplemented

    @staticmethod
    def from_string(data: str) -> "Antenna":

        content = data.split()[1:]
        if len(content) != 21:
            log.error(
                "Failed to initialize Antenna: String contains invalid number of "
                f"parameters ({len(content)})"
            )
            exit(1)

        _input: Any = content[:5] + [float(x) for x in content[5:]]

        # Turn radome flag into boolean
        _input[3] = True if _input[3] == "RA_YES" else False

        return Antenna(*_input)

    def __getattribute__(self, name: str) -> Any:

        val = super().__getattribute__(name)
        if val is NotImplemented:
            raise NotImplementedError(f"Attribute {name} is not initialized")
        return val
