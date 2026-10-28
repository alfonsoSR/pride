from pathlib import Path
from astropy import time
import numpy as np
from ...logger import log
from ... import utils


def write_doppler_file(
    doppler_file: str | Path,
    header: list[str],
    epochs: "time.Time",
    sky_frequency: np.ndarray,
    corrections: list[np.ndarray],
) -> None:
    """Write Doppler predictions to a text file

    Each line of the file contains, separated by spaces:
    1. UTC epoch of reception in ISO 8601 format
    2. Modified Julian Date (integer part)
    3. UTC seconds past midnight
    4. Ephemeris time: TDB seconds past J2000
    5. Sky frequency at the receiving station [Hz]
    6+. One column for each correction [Hz] (DopplerPredictor writes: troposphere,
    ionosphere, solid tide, ocean loading, pole tide, antenna)

    The header lines are written at the beginning of the file, preceded by '#'.

    :param doppler_file: Path to the output file
    :param header: Lines of the header (without leading '#')
    :param epochs: UTC epochs of reception (N,)
    :param sky_frequency: Sky frequency at the receiving station (N,)
    :param corrections: List of corrections, each as (N,) array
    """

    doppler_file = Path(doppler_file).resolve()
    if doppler_file.exists():
        log.warning(f"Overwriting existing Doppler file: {doppler_file}")

    # Time columns
    utc = epochs.utc  # type: ignore
    iso_epochs = time.Time(utc, precision=6).isot
    mjd = np.floor(utc.mjd)  # type: ignore
    seconds_past_midnight = np.array(
        [
            dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond * 1e-6
            for dt in utc.to_datetime()
        ]
    )
    et = utils.get_ephemeris_time_from_epoch(utc)

    with doppler_file.open("w") as f:

        for line in header:
            f.write(f"# {line}\n")

        for idx in range(len(sky_frequency)):
            f.write(
                f"{iso_epochs[idx]} {mjd[idx]:.3f} "
                f"{seconds_past_midnight[idx]:.10f} {et[idx]:.6f} "
                f"{sky_frequency[idx]:23.16e}"
                + "".join(
                    f" {correction[idx]:14.7e}" for correction in corrections
                )
                + "\n"
            )

    log.info(f"Succesfully generated {doppler_file}")

    return None
