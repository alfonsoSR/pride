import numpy as np
from astropy import units


def eops_arcsec2rad(eops: np.ndarray) -> np.ndarray:
    """Convert EOPs from arcsec to rad

    :param eops: EOPs in s, arcsec
    :return: EOPs in s, rad
    """
    # Convert EOPs to s and rad [Original units are s and arcsec]
    ut1_utc, xp_as, yp_as, dx_as, dy_as = eops
    xp, yp, dx, dy = (
        units.Quantity([xp_as, yp_as, dx_as, dy_as], "arcsec").to("rad").value
    )
    return np.array([ut1_utc, xp, yp, dx, dy])
