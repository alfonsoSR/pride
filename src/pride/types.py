"""Containers and auxiliary data types meant to be used internally"""

from dataclasses import dataclass
from .logger import log
from typing import Literal, Any, TYPE_CHECKING, Mapping
from enum import Enum
import numpy as np
from .io import load_catalog

if TYPE_CHECKING:
    from .io import Setup


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


class SourceType(Enum):
    """Types of observable sources

    Defines the model used to compute geometric delays and the applicability of Doppler corrections
    """

    FarField = 1
    NearField = 2


class Constants:
    """Physical constants for a specific version of the JPL ephemerides

    NOTE: Conversions between TDB, TCB and TT
    mass_TCB = mass_TDB * (1 + L_B)
    mass_TT = mass_TCB * (1 - L_C)
    """

    VALID_VERSIONS = Literal["421", "405", "403", "430", "440", "13c"]

    def __init__(self, jpl_eph: VALID_VERSIONS) -> None:
        """COPY PASTED FROM ORIGINAL CODE!"""

        # jpl_eph used can be 405 or 421, the latter is the default
        # math consts
        self.CDEGRAD = 1.7453292519943296e-02
        self.CARCRAD = 4.8481368110953599e-06
        self.CTIMRAD = 7.2722052166430399e-05
        self.SECDAY = 86400.0
        self.JUL_CENT = 36525.0
        # JULIAN DATE OF STANDARD EPOCH J2000.0
        self.JD2000 = 2451545.0
        # Algemeene Physical constants
        self.C = 2.99792458e8
        self.C_km = 2.99792458e5
        self.F = 298.25642  # the tide-free value, IERS2010
        self.AE = 6378136.3  # the tide-free value
        self.J_2 = 1.0826e-3
        self.AU = 149597870700.0  # DE430
        self.TAUA = 499.0047838061
        self.G = 6.67428e-11

        # from tn36; also see http://ilrs.gsfc.nasa.gov/docs/2014/196C.pdf
        self.L_B = 1.550519768e-8
        self.L_C = 1.48082686741e-8
        self.L_G = 6.969290134e-10

        # DE/LE405 Header. TDB-compatible!!
        if "403" in jpl_eph:
            AU_DE405 = 1.49597870691000015e11  # m
            self.GSUN = 0.295912208285591095e-03 * AU_DE405**3 / 86400.0**2
            self.MU = 0.813005600000000044e02 ** (-1)
            self.GEARTH = (
                0.899701134671249882e-09
                * AU_DE405**3
                / 86400.0**2
                / (1 + self.MU)
            )
            self.GMOON = self.GEARTH * self.MU
            self.GMPlanet = [
                0.491254745145081187e-10 * AU_DE405**3 / 86400.0**2,
                0.724345248616270270e-09 * AU_DE405**3 / 86400.0**2,
                0.954953510577925806e-10 * AU_DE405**3 / 86400.0**2,
                0.282534590952422643e-06 * AU_DE405**3 / 86400.0**2,
                0.845971518568065874e-07 * AU_DE405**3 / 86400.0**2,
                0.129202491678196939e-07 * AU_DE405**3 / 86400.0**2,
                0.152435890078427628e-07 * AU_DE405**3 / 86400.0**2,
                0.218869976542596968e-11 * AU_DE405**3 / 86400.0**2,
            ]
        elif "405" in jpl_eph:
            AU_DE405 = 1.49597870691000015e11  # m
            self.GSUN = 0.295912208285591095e-03 * AU_DE405**3 / 86400.0**2
            self.MU = 0.813005600000000044e02 ** (-1)
            self.GEARTH = (
                0.899701134671249882e-09
                * AU_DE405**3
                / 86400.0**2
                / (1 + self.MU)
            )
            self.GMOON = self.GEARTH * self.MU
            self.GMPlanet = [
                0.491254745145081187e-10 * AU_DE405**3 / 86400.0**2,
                0.724345248616270270e-09 * AU_DE405**3 / 86400.0**2,
                0.954953510577925806e-10 * AU_DE405**3 / 86400.0**2,
                0.282534590952422643e-06 * AU_DE405**3 / 86400.0**2,
                0.845971518568065874e-07 * AU_DE405**3 / 86400.0**2,
                0.129202491678196939e-07 * AU_DE405**3 / 86400.0**2,
                0.152435890078427628e-07 * AU_DE405**3 / 86400.0**2,
                0.218869976542596968e-11 * AU_DE405**3 / 86400.0**2,
            ]
        # DE/LE421 Header. TDB-compatible!!
        elif "421" in jpl_eph:
            AU_DE421 = 1.49597870699626200e11  # m
            self.GSUN = 0.295912208285591100e-03 * AU_DE421**3 / 86400.0**2
            self.MU = 0.813005690699153000e02 ** (-1)
            self.GEARTH = (
                0.899701140826804900e-09
                * AU_DE421**3
                / 86400.0**2
                / (1 + self.MU)
            )
            self.GMOON = self.GEARTH * self.MU
            self.GMPlanet = [
                0.491254957186794000e-10 * AU_DE421**3 / 86400.0**2,
                0.724345233269844100e-09 * AU_DE421**3 / 86400.0**2,
                0.954954869562239000e-10 * AU_DE421**3 / 86400.0**2,
                0.282534584085505000e-06 * AU_DE421**3 / 86400.0**2,
                0.845970607330847800e-07 * AU_DE421**3 / 86400.0**2,
                0.129202482579265000e-07 * AU_DE421**3 / 86400.0**2,
                0.152435910924974000e-07 * AU_DE421**3 / 86400.0**2,
                0.217844105199052000e-11 * AU_DE421**3 / 86400.0**2,
            ]
        # DE/LE430 Header. TDB-compatible!!
        elif "430" in jpl_eph:
            AU_DE430 = 1.49597870700000000e11  # m
            self.GSUN = 0.295912208285591100e-03 * AU_DE430**3 / 86400.0**2
            self.MU = 0.813005690741906200e02 ** (-1)
            self.GEARTH = (
                0.899701139019987100e-09
                * AU_DE430**3
                / 86400.0**2
                / (1 + self.MU)
            )
            self.GMOON = self.GEARTH * self.MU
            self.GMPlanet = [
                0.491248045036476000e-10 * AU_DE430**3 / 86400.0**2,
                0.724345233264412000e-09 * AU_DE430**3 / 86400.0**2,
                0.954954869555077000e-10 * AU_DE430**3 / 86400.0**2,
                0.282534584083387000e-06 * AU_DE430**3 / 86400.0**2,
                0.845970607324503000e-07 * AU_DE430**3 / 86400.0**2,
                0.129202482578296000e-07 * AU_DE430**3 / 86400.0**2,
                0.152435734788511000e-07 * AU_DE430**3 / 86400.0**2,
                0.217844105197418000e-11 * AU_DE430**3 / 86400.0**2,
            ]
        elif "440" in jpl_eph:
            """https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440_tech-comments.txt"""
            AU_DE440 = 1.49597870700000000e11  # m
            self.GSUN = 0.295912208284119560e-03 * AU_DE440**3 / 86400**2
            self.MU = 0.813005682214972e02 ** (-1)
            self.GEARTH = (
                0.899701139294734660e-09
                * AU_DE440**3
                / 86400**2
                / (1 + self.MU)
            )
            self.GMOON = self.GEARTH * self.MU
            self.GMPlanet = [
                0.491250019488931820e-10 * AU_DE440**3 / 86400**2,
                0.724345233264411870e-09 * AU_DE440**3 / 86400**2,
                0.954954882972581190e-10 * AU_DE440**3 / 86400**2,
                0.282534582522579170e-06 * AU_DE440**3 / 86400**2,
                0.845970599337629030e-07 * AU_DE440**3 / 86400**2,
                0.129202656496823990e-07 * AU_DE440**3 / 86400**2,
                0.152435734788519390e-07 * AU_DE440**3 / 86400**2,
                0.217509646489335810e-11 * AU_DE440**3 / 86400**2,
            ]
        # INPOP13c Header. TDB-compatible!!
        elif "13c" in jpl_eph:
            AU_13c = 1.495978707000000e11
            self.GSUN = 0.2959122082912712e-03 * (AU_13c**3) / (86400.0**2)
            self.MU = 0.8130056945994197e02 ** (-1)
            self.GEARTH = (
                0.8997011572788968e-09 * AU_13c**3 / 86400.0**2 / (1 + self.MU)
            )
            self.GMOON = self.GEARTH * self.MU
            self.GMPlanet = [
                0.4912497173300158e-10 * AU_13c**3 / 86400.0**2,
                0.7243452327305554e-09 * AU_13c**3 / 86400.0**2,
                0.9549548697395966e-10 * AU_13c**3 / 86400.0**2,
                0.2825345791109909e-06 * AU_13c**3 / 86400.0**2,
                0.8459705996177680e-07 * AU_13c**3 / 86400.0**2,
                0.1292024916910406e-07 * AU_13c**3 / 86400.0**2,
                0.1524357330444817e-07 * AU_13c**3 / 86400.0**2,
                0.2166807318808926e-11 * AU_13c**3 / 86400.0**2,
            ]
        else:
            raise ValueError(
                f"Version {jpl_eph} of JPL ephemeris is not supported"
            )

        self.TDB_TCB = 1.0 + self.L_B  # F^-1
        # G*masses in TCB-frame!
        self.GM_TCB = (
            np.hstack(
                [
                    self.GMPlanet[0:2],
                    self.GEARTH,
                    self.GMPlanet[2:],
                    self.GMOON,
                    self.GSUN,
                ]
            )
            * self.TDB_TCB
        )
        # G*masses in TDB-frame!
        self.GM = np.hstack(
            [
                self.GMPlanet[0:2],
                self.GEARTH,
                self.GMPlanet[2:],
                self.GMOON,
                self.GSUN,
            ]
        )

        return None

    @staticmethod
    def from_setup(setup: "Setup") -> "Constants":

        # Get path to meta-kernel file
        __catalog = load_catalog("spacecraft.yaml")
        __sc = str(setup.general["target"]).upper()
        target: dict[str, Any] | None = None
        for _target in __catalog.values():
            if __sc in _target["names"]:
                target = _target
                break
        if target is None:
            raise ValueError("Specified target not in catalog: spacecraft.yaml")

        kernel_path = setup.resources["ephemerides"] / target["short_name"]
        metak_path = kernel_path / "metak.tm"

        # Get version of ephemerides from meta-kernel
        log.debug("LACKING ROBUST WAY TO GET VERSION OF EPHEMERIDES")
        version: str = ""
        with metak_path.open() as f:
            for line in f:
                if "spk/DE" in line and ".BSP" in line:
                    version = line.split("/")[-1].split(".")[0]
                    break
        if version == "":
            raise ValueError(
                "Failed to read ephemerides version from meta-kernel"
            )

        if version not in Constants.VALID_VERSIONS.__args__:
            raise ValueError(
                f"Version {version} of JPL ephemeris is not supported"
            )

        return Constants(version)


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
