import numpy as np
from ...constants import CLIGHT


TURNAROUND_RATIOS: dict[tuple[str, str], float] = {
    ("S", "S"): 240.0 / 221.0,
    ("S", "X"): 880.0 / 221.0,
    ("S", "KA"): 3344.0 / 221.0,
    ("X", "S"): 240.0 / 749.0,
    ("X", "X"): 880.0 / 749.0,
    ("X", "KA"): 3344.0 / 749.0,
    ("KA", "S"): 240.0 / 3599.0,
    ("KA", "X"): 880.0 / 3599.0,
    ("KA", "KA"): 3344.0 / 3599.0,
}
"""Turnaround ratios for (uplink, downlink) bands [DSN 810-005 Rev. E, module 201, Table 1]

The Ka-band ratios are the recommended ones; existing Ka-band spacecraft have used ratios negotiated with the JPL Frequency Manager (e.g. 3360/749 for X/Ka), which are not included.
"""


def get_turnaround_ratio(uplink_band: str, downlink_band: str) -> float | None:
    """Turnaround ratio of a transponder for a pair of frequency bands

    :param uplink_band: Uplink band ("S", "X" or "Ka"; case-insensitive)
    :param downlink_band: Downlink band ("S", "X" or "Ka"; case-insensitive)
    :return: Ratio of downlink and uplink frequencies, or None if not available
    """

    return TURNAROUND_RATIOS.get(
        (str(uplink_band).upper(), str(downlink_band).upper())
    )


def calculate_frequency_ratio(
    bodies_gm: np.ndarray,
    x_tx_bcrf: np.ndarray,
    v_tx_bcrf: np.ndarray,
    x_bodies_bcrf_tx: np.ndarray,
    v_bodies_bcrf_tx: np.ndarray,
    x_rx_bcrf: np.ndarray,
    v_rx_bcrf: np.ndarray,
    x_bodies_bcrf_rx: np.ndarray,
    v_bodies_bcrf_rx: np.ndarray,
) -> np.ndarray:
    """Ratio of received and transmitted frequencies for a single leg

    Calculates the ratio between the proper frequency of a signal at the receiver and its proper frequency at the transmitter, for a signal travelling along one leg of a radio link (e.g. spacecraft to station, or station to spacecraft). The model accounts for:
    - The rates of the proper times of transmitter and receiver with respect to TCB, due to their BCRF velocities and the Newtonian potential of the massive bodies.
    - The first-order (classical) Doppler effect along the line of sight.
    - The gravitational frequency shift, obtained as the time derivative of the Shapiro delay caused by each massive body.

    For an N-way link (e.g. three-way), the total ratio is the product of the ratios of the individual legs.

    The algorithm follows the PhD thesis of D. Duev (MSU, 2012) and its original implementation in pypride (`doppler_bc`). Unlike the original implementation, the numerator of the gravitational frequency shift uses the exact derivative of the Shapiro delay, in which the term 2GM/c^2 scales the rate of change of the distance between transmitter and receiver.

    In the description of the arguments, M is the number of massive bodies, N is the number of epochs, and the tuples indicate the expected shape of the arrays.

    Source: Duev, D. A. et al. (2012). Spacecraft VLBI and Doppler tracking: Algorithms and implementation. A&A 541, A43. https://doi.org/10.1051/0004-6361/201218885

    :param bodies_gm: Gravitational parameters of the massive bodies (M,)
    :param x_tx_bcrf: BCRF position of the transmitter at TX epochs (N, 3)
    :param v_tx_bcrf: BCRF velocity of the transmitter at TX epochs (N, 3)
    :param x_bodies_bcrf_tx: BCRF position of the massive bodies at TX epochs (M, N, 3)
    :param v_bodies_bcrf_tx: BCRF velocity of the massive bodies at TX epochs (M, N, 3)
    :param x_rx_bcrf: BCRF position of the receiver at RX epochs (N, 3)
    :param v_rx_bcrf: BCRF velocity of the receiver at RX epochs (N, 3)
    :param x_bodies_bcrf_rx: BCRF position of the massive bodies at RX epochs (M, N, 3)
    :param v_bodies_bcrf_rx: BCRF velocity of the massive bodies at RX epochs (M, N, 3)
    :return: Ratio between received and transmitted proper frequencies (N,)
    """

    # Precomputed quantities
    clight2 = CLIGHT * CLIGHT
    gm = bodies_gm[:, None]  # (M, 1)
    gmc = 2.0 * gm / clight2  # (M, 1)

    # Positions and velocities relative to the massive bodies
    x_tx_bodies = x_tx_bcrf[None, :, :] - x_bodies_bcrf_tx  # (M, N, 3)
    x_rx_bodies = x_rx_bcrf[None, :, :] - x_bodies_bcrf_rx  # (M, N, 3)
    v_tx_bodies = v_tx_bcrf[None, :, :] - v_bodies_bcrf_tx  # (M, N, 3)
    v_rx_bodies = v_rx_bcrf[None, :, :] - v_bodies_bcrf_rx  # (M, N, 3)
    r_tx_bodies = np.linalg.norm(x_tx_bodies, axis=-1)  # (M, N)
    r_rx_bodies = np.linalg.norm(x_rx_bodies, axis=-1)  # (M, N)

    # Relative position of receiver wrt transmitter, as seen from the bodies
    x_txrx_bodies = x_rx_bodies - x_tx_bodies  # (M, N, 3)
    r_txrx_bodies = np.linalg.norm(x_txrx_bodies, axis=-1)  # (M, N)

    # Rates of change of the distances involved in the Shapiro delay
    dot_r_txrx = (
        np.sum(x_txrx_bodies * (v_rx_bodies - v_tx_bodies), axis=-1)
        / r_txrx_bodies
    )  # (M, N)
    dot_r_sum = (
        np.sum(x_rx_bodies * v_rx_bodies, axis=-1) / r_rx_bodies
        + np.sum(x_tx_bodies * v_tx_bodies, axis=-1) / r_tx_bodies
    )  # (M, N)
    r_sum = r_rx_bodies + r_tx_bodies  # (M, N)

    # Gravitational frequency shift: Minus time derivative of Shapiro delay
    z_sh = -np.sum(
        (2.0 * gmc / CLIGHT)
        * (dot_r_txrx * (r_sum + gmc) - r_txrx_bodies * dot_r_sum)
        / ((r_sum + gmc) ** 2 - r_txrx_bodies**2),
        axis=0,
    )  # (N,)

    # Rates of proper time of transmitter and receiver wrt TCB
    dtau_tx_dtcb = 1.0 - (
        np.sum(gm / r_tx_bodies, axis=0)
        + 0.5 * np.sum(v_tx_bcrf * v_tx_bcrf, axis=-1)
    ) / clight2
    dtau_rx_dtcb = 1.0 - (
        np.sum(gm / r_rx_bodies, axis=0)
        + 0.5 * np.sum(v_rx_bcrf * v_rx_bcrf, axis=-1)
    ) / clight2

    # Unit vector from receiver to transmitter
    x_rx_tx = x_tx_bcrf - x_rx_bcrf
    n_rx_tx = x_rx_tx / np.linalg.norm(x_rx_tx, axis=-1)[:, None]

    # First-order Doppler effect
    doppler_rx = 1.0 + np.sum(n_rx_tx * v_rx_bcrf, axis=-1) / CLIGHT
    doppler_tx = 1.0 + np.sum(n_rx_tx * v_tx_bcrf, axis=-1) / CLIGHT

    return (1.0 + z_sh) * dtau_tx_dtcb * doppler_rx / (dtau_rx_dtcb * doppler_tx)
