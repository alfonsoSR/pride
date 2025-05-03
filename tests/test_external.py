import pytest
from pride.external import iers
import numpy as np


def test_dehanttideinel() -> None:

    xsta = np.array([1112200.5696e0, -4842957.8511e0, 3985345.9122e0])
    xsun = np.array([100210282451.6279e0, 103055630398.3160e0, 56855096480.4475e0])
    xmoon = np.array([369817604.4348e0, 1897917.5258e0, 120804980.8284e0])
    yr = 2015
    month = 7
    day = 15
    fhr = 0.00e0

    expected = np.array(
        [0.00509570869172363845e0, 0.0828663025983528700e0, -0.0636634925404189617e0]
    )
    dxtide = iers.dehanttideinel(xsta, yr, month, day, fhr, xsun, xmoon)

    diff = dxtide - expected
    assert np.allclose(diff, 0, atol=1e-15)

    return None
