import numpy as np
from astropy.utils import iers
from astropy import time, units
from typing import Literal, TYPE_CHECKING
from ..logger import log
from ..external.iers import interp
import erfa
from .. import utils, math

if TYPE_CHECKING:
    from ..experiment.experiment import Experiment


class EOP:
    """Earth Orientation Parameters"""

    def __init__(
        self,
        bulletin: Literal["A", "B"],
        initial_epoch: time.Time,
        final_epoch: time.Time,
        margin_in_days: int = 8,
    ) -> None:
        """Constructor for EOP interface class

        .. Note:: You can find more information about EOP bulletins at the API documentation of this module

        :param bulletin: EOP bulletin to use [A or B]
        :param initial_epoch: Epoch from which to load EOPs
        :param final_epoch: Epoch until which to load EOPs
        :param margin_in_days: Number of days by which to augment the time range defined by the initial and final epochs.
        """

        # Select EOP bulletin
        match bulletin:
            case "A":
                eop_table = iers.IERS_A.open()
            case "B":
                eop_table = iers.IERS_B.open()
            case _:
                log.error(
                    f"Failed to initialize EOPs: Invalid bulletin {bulletin}"
                )
                exit(1)

        # Augment initial and final epochs with margin
        initial_mjd = (initial_epoch.mjd // 1) - margin_in_days  # type: ignore
        final_mjd = (final_epoch.mjd // 1) + margin_in_days  # type: ignore
        self.__mjd_range = (initial_mjd, final_mjd)

        # Find entries of EOP table within the time range
        eop_table_mask: np.ndarray = (
            np.argwhere(
                (eop_table["MJD"].data >= initial_mjd)  # type: ignore
                & (eop_table["MJD"].data <= final_mjd)  # type: ignore
            )
            .ravel()
            .astype(int)
        )
        assert isinstance(eop_table_mask, np.ndarray)

        # Get MJD for table entries in the time range
        relevant_mjds = eop_table["MJD"][eop_table_mask].data  # type: ignore

        # EOPs at integer epochs
        self.__eops_dict = {
            "xp": math.Default1DInterpolator(
                relevant_mjds,
                eop_table["PM_x"][eop_table_mask].value,  # type: ignore
            ),
            "yp": math.Default1DInterpolator(
                relevant_mjds,
                eop_table["PM_y"][eop_table_mask].value,  # type: ignore
            ),
            "ut1_utc": math.Default1DInterpolator(
                relevant_mjds,
                eop_table["UT1_UTC"][eop_table_mask].value,  # type: ignore
            ),
            "dx": math.Default1DInterpolator(
                relevant_mjds,
                eop_table["dX_2000A"][eop_table_mask].value,  # type: ignore
            ),
            "dy": math.Default1DInterpolator(
                relevant_mjds,
                eop_table["dY_2000A"][eop_table_mask].value,  # type: ignore
            ),
        }

        return None

    def __eops(self, mjds) -> np.ndarray:

        if np.any(mjds < self.__mjd_range[0]) or np.any(
            mjds > self.__mjd_range[1]
        ):
            log.error("Requested epoch is out of range")
            exit(1)

        return np.array(
            [
                self.__eops_dict["xp"](mjds),
                self.__eops_dict["yp"](mjds),
                self.__eops_dict["ut1_utc"](mjds),
                self.__eops_dict["dx"](mjds),
                self.__eops_dict["dy"](mjds),
            ]
        )

    @staticmethod
    def from_experiment(experiment: "Experiment") -> "EOP":

        raise DeprecationWarning(
            "Initializing EOPs from experiment is deprecated"
        )
        return EOP(
            experiment.setup.internal["eop_bulletin"],
            experiment.initial_epoch,
            experiment.final_epoch,
        )

    def at_epoch(
        self, epoch: "time.Time", unit: Literal["rad", "arcsec"] = "arcsec"
    ) -> np.ndarray:
        """Interpolate EOPs at given epoch

        Returns interpolated EOPs corrected for oceanic tides and libration effects. Corrections are computed using a the INTERP.F program, described in section 5.5.1 of the IERS Conventions (2010) and available at https://hpiers.obspm.fr/eop-pc/index.php?index=FTP&lang=en

        :param epoch: UTC epochs at which to interpolate EOPs
        :param unit: Units of output EOPs. Options are 'arcsec' and 'rad'
        :return: Interpolated EOPs as a (5, N) array. The order of EOPs is UT1-UTC, X, Y, dX, dY
        """

        if epoch.isscalar:
            epoch = time.Time(
                [
                    epoch,
                ]
            )

        # Get MJDs for given epochs
        mjd_int_list: np.ndarray = np.array([epoch.mjd // 1], dtype=int).ravel()  # type: ignore
        mjd_list = np.array(epoch.mjd)

        # Interpolate with corrections
        out = np.zeros((len(mjd_int_list), 5))
        for idx, (mjd_int, mjd) in enumerate(zip(mjd_int_list, mjd_list)):

            # Define a range of seven days around epoch
            mjd_range = np.arange(mjd_int - 3, mjd_int + 4)

            # Get default values of EOPs for range of epochs
            ut1_utc, xp, yp, dx, dy = self.__eops(mjd_range)

            # Interpolate X, Y and UT1-UTC
            x, y, ut1_utc = interp.interp(
                mjd_range, xp, yp, ut1_utc, rjd_int=mjd
            )

            # Interpolate dX and dY
            dx = interp.lagint(mjd_range, dx, xint=mjd)
            dy = interp.lagint(mjd_range, dy, xint=mjd)

            # Store interpolated values
            out[idx] = np.array([ut1_utc, x, y, dx, dy])

        # Return in requested units
        match unit:
            case "arcsec":
                return out.T
            case "rad":
                x, y, dx, dy = (
                    units.Quantity(out.T[1:], "arcsec").to("rad").value
                )
                return np.array([out.T[0], x, y, dx, dy]).T
            case _:
                log.error(f"Failed to generate EOPs: Invalid unit {unit}")
                exit(1)


def seu2itrf(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Rotation matrix from SEU to ITRF

    :param lat: Geodetic latitude in radians
    :param lon: Geodetic longitude in radians
    :return: Rotation matrix from SEU to ITRF as (N, 3, 3) array
    """
    return erfa.rz(-lon, erfa.ry(-lat, np.eye(3)))


def icrf2itrf(eops: np.ndarray, epoch: "time.Time") -> np.ndarray:
    """Rotation matrix from ICRF to ITRF

    :param eop: Earth Orientation Parameters
    :param epoch: UTC epochs at which to compute matrix
    :return: Rotation matrix from ICRF to ITRF as (N, 3, 3) array
    """

    # Epochs in TT and UTC
    tt_epoch: time.Time = epoch.tt  # type: ignore
    utc_epoch: time.Time = epoch.utc  # type: ignore

    # Get EOPs in s and rad
    ut1_utc, xp, yp, dx, dy = utils.eops_arcsec2rad(eops)

    # Compute matrix following algorithm from IERS Conventions 2010 (5.9)
    x, y = erfa.xy06(tt_epoch.jd1, tt_epoch.jd2)
    s = erfa.s06(tt_epoch.jd1, tt_epoch.jd2, x, y)
    RC2I_matrix = erfa.c2ixys(x + dx, y + dy, s)
    era = erfa.era00(utc_epoch.jd1, utc_epoch.jd2 + (ut1_utc / 86400.0))
    RC2TI_matrix = erfa.rz(era, RC2I_matrix)
    s_prime = erfa.sp00(tt_epoch.jd1, tt_epoch.jd2)
    RPOM_matrix = erfa.pom00(xp, yp, s_prime)

    return RPOM_matrix @ RC2TI_matrix


def itrf2icrf(eops: np.ndarray, epoch: "time.Time") -> np.ndarray:
    """Rotation matrix from ITRF to ICRF

    :param eop: Earth Orientation Parameters
    :param epoch: UTC epochs at which to compute matrix
    :return: Rotation matrix from ITRF to ICRF as (N, 3, 3) array
    """

    return icrf2itrf(eops, epoch).swapaxes(-1, -2)
