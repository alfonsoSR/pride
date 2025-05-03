def vmf3(
    ah: float,
    aw: float,
    mjd: float,
    lat: float,
    lon: float,
    zd: float,
) -> tuple[float, float]:
    """Calculate VMF3 mapping factors from site-wise data

    This is a wrapper around the 'vmf3' subroutine defined in the 'vmf3.f90'
    script of the VMF3 model. As of 24/02/2025, it can be downloaded from
    https://vmf.geo.tuwien.ac.at/codes/vmf3.f90

    Original docstring
    ------------------
    (c) Department of Geodesy and Geoinformation, Vienna University of
    Technology, 2016

    This subroutine determines the VMF3 hydrostatic and wet mapping factors.
    The a coefficients have to be inserted from discrete data, while the b
    and c coefficients are of empirical nature containing a geographical
    and temporal dependence, represented in spherical harmonics. The
    spherical harmonics coefficients are developed to degree and order 12 and
    are based on a 5°x5° grid containing ray-tracing data from 2001-2010.
    All input quantities have to be scalars!

    :param ah: Hydrostatic mapping function coefficient 'a'
    :param aw: Wet mapping function coefficient 'a'
    :param mjd: Modified Julian Date
    :param lat: Geodetic latitude of the site (GRS80) [rad]
    :param lon: Geodetic longitude of the site (GRS80) [rad]
    :param zd: Zenith distance (0.5 * pi - elevation) [rad]
    """
    ...
