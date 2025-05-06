import numpy as np

def interp(
    jd: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ut1_utc: np.ndarray,
    n: int | None = None,
    rjd_int: float = 0.0,
) -> tuple[float, float, float]:
    """Interpolate EOPs

    :param jd: Epochs for which the EOPs are known as julian days
    :param x: Pole offset in x direction
    :param y: Pole offset in y direction
    :param ut1_utc: UT1-UTC
    :param n: Length of input arrays [DO NOT USE]
    :param rjd_int: Epoch for which to interpolate the EOPs
    :return: Interpolated EOPs [x, y, ut1_utc]
    """
    ...

def lagint(
    x: np.ndarray, y: np.ndarray, n: int | None = None, xint: float = 0
) -> float:
    """Lagrange interpolation

    :param x: Array of x values
    :param y: Array of y values
    :param n: Length of input arrays
    :param xint: Value at which to interpolate
    :return: Interpolated value
    """
    ...
