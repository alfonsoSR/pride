import numpy as np
from ...constants import CLIGHT


def post_newtonian_near_field_effect(
    bodies_gm: np.ndarray,
    x_obs_bcrf_rx: np.ndarray,
    x_src_bcrf_tx: np.ndarray,
    x_bodies_bcrf_rx: np.ndarray,
    x_bodies_bcrf_tx: np.ndarray,
) -> None:
    """Post-Newtonian gravitational effect due to moving solar system bodies

    source: duev 2012

    :param bodies_gm: Standard gravitational parameter of all the massive bodies to be considered. Array of shape (M,), with M the number of bodies.
    :param x_obs_bcrf_rx: BCRF position vector of the observer at RX epochs. Array of shape (N, 3), with N the number of epochs.
    :param x_src_bcrf_tx: BCRF position vector of the source at TX epochs. Array of shape (N, 3), with N the number of epochs.
    :param x_bodies_bcrf_rx: BCRF position vectors of the solar system bodies at RX epochs. Array of shape (M, N, 3), with M the number of bodies and N the number of epochs.
    :param x_bodies_bcrf_tx: BCRF position vectors of the solar system bodies at TX epochs. Array of shape (M, N, 3), with M the number of bodies and N the number of epochs.
    :return: Post-Newtonian effect on light travel time between source and observer. Array of shape (N,), with N the number of epochs.
    """

    # Relative position of source and bodies at TX
    x_src_bodies_bcrf = (
        x_src_bcrf_tx[None, :, :] - x_bodies_bcrf_tx
    )  # (M, N, 3)
    r_src_bodies = np.linalg.norm(x_src_bodies_bcrf, axis=-1)  # (M, N)
    r0b = x_src_bodies_bcrf
    r0b_mag = r_src_bodies

    # Relative position of observer and bodies at RX
    x_obs_bodies_bcrf = (
        x_obs_bcrf_rx[None, :, :] - x_bodies_bcrf_rx
    )  # (M, N, 3)
    r_obs_bodies = np.linalg.norm(x_obs_bodies_bcrf, axis=-1)  # (M, N)
    r1b = x_obs_bodies_bcrf
    r1b_mag = r_obs_bodies

    # Difference of relative positions with respect to the bodies
    r01b_mag = np.linalg.norm(r1b - r0b, axis=-1)  # (M, N)

    # Reused term: 2 * GM / c^2
    gmc = 2.0 * bodies_gm[:, None] / (CLIGHT * CLIGHT)  # (M, 1)

    # Post-Newtonian effect
    dt_pn = np.sum(
        (gmc / CLIGHT)
        * np.log(
            (r0b_mag + r1b_mag + r01b_mag + gmc)
            / (r0b_mag + r1b_mag - r01b_mag + gmc)
        ),
        axis=0,
    )  # (N,)

    return dt_pn
