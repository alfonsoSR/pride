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
