import spiceypy as spice
import numpy as np
from ..logger import log


def get_icrf_state_vector(target: str, epochs: np.ndarray) -> np.ndarray:
    """Get cartesian state vector of a target in ICRF (SI units)

    Uses spice.spkerz to get the state of a target in J2000, with respect to SSB, and with aberrations set to NONE. The output is transformed to SI units and casted to an (N, 6) numpy array.

    :param target: Target name known by spice
    :param epochs: Array of epochs in ET (TDB seconds past J2000)
    :return state_vector: Aberrated, cartesian state vector of the target with respect to SSB in J2000 (ICRF) frame.
    """

    # Get state and LT from spice
    cstate_tuple, _ = spice.spkezr(target, epochs, "J2000", "NONE", "SSB")

    # Convert to meters and numpy array
    return np.array(cstate_tuple, dtype=np.float64) * 1e3


def get_icrf_position_vector(target: str, epochs: np.ndarray) -> np.ndarray:
    """Get cartesian position vector of a target in ICRF (SI units)

    Uses spice.spkpos to get the position of a target in J2000, with respect to SSB, and with aberrations set to NONE. The output is transformed to SI units and casted to an (N, 3) numpy array.

    :param target: Target name known by spice
    :param epochs: Array of epochs in ET (TDB seconds past J2000)
    :return position_vector: Aberrated, cartesian position vector of the target with respect to SSB in J2000 (ICRF) frame.
    """

    # Get position and LT from spice
    cpos_tuple, _ = spice.spkpos(target, epochs, "J2000", "NONE", "SSB")

    # Convert to meters and numpy array
    return np.array(cpos_tuple, dtype=np.float64) * 1e3


def get_gcrf_position_vector(target: str, epochs: np.ndarray) -> np.ndarray:
    """Get cartesian position vector of a target in GCRF (SI units)

    Uses spice.spkpos to get the position of a target in J2000, with respect to EARTH, and with aberrations set to NONE. The output is transformed to SI units and casted to an (N, 3) numpy array.

    :param target: Target name known by spice
    :param epochs: Array of epochs in ET (TDB seconds past J2000)
    :return position_vector: Aberrated, cartesian position vector of the target with respect to EARTH in J2000 (GCRF) frame.
    """

    # Get position and LT from spice
    cpos_tuple, _ = spice.spkpos(target, epochs, "J2000", "NONE", "EARTH")

    # Convert to meters and numpy array
    return np.array(cpos_tuple, dtype=np.float64) * 1e3


def get_gcrf_state_vector(target: str, epochs: np.ndarray) -> np.ndarray:
    """Get cartesian state vector of a target in GCRF (SI units)

    Uses spice.spkezr to get the position of a target in J2000, with respect to EARTH, and with aberrations set to NONE. The output is transformed to SI units and casted to an (N, 6) numpy array.

    :param target: Target name known by spice
    :param epochs: Array of epochs in ET (TDB seconds past J2000)
    :return position_vector: Aberrated, cartesian state vector of the target with respect to EARTH in J2000 (GCRF) frame.
    """

    # Get position and LT from spice
    cstate_tuple, _ = spice.spkezr(target, epochs, "J2000", "NONE", "EARTH")

    # Convert to meters and numpy array
    return np.array(cstate_tuple, dtype=np.float64) * 1e3


def get_body_gravitational_parameter(target: str) -> float:
    """Get gravitational parameter of target from SPICE kernels

    Uses spice.bodvrd to retrieve the gravitational parameter of the target in the loaded version of the SPICE kernels. The result is transformed to SI units.

    :param target: Target name known to SPICE
    :return mu_target: Gravitational parameter of the target in SI units
    """

    return spice.bodvrd(target, "GM", 1)[1][0] * 1e9
