from astropy import time
import numpy as np
from ..constants import CLIGHT
from .. import utils
from . import ephemerides, relativity


def light_time_from_rx_epoch(
    tdb_rx: "time.Time",
    xrec_bcrf_rx: np.ndarray,
    source_id: str,
    perturbing_bodies: list[str],
    precision: float,
    max_iterations: int,
) -> "time.TimeDelta":
    """Light-time from source to receiver

    Given a reception epoch and the associated position of the receiver in BCRF,
    calculates the light-time for a source (transmitter) identified by a name
    or ID known to SPICE, using the Newton-Raphson algorithm as formulated in Duev 2012 (10.1051/0004-6361/201218885).

    The estimated light-time results from iteratively solving the light-time
    equation with precision settings specified by the user, and considering
    post-Newtonian relativistic perturbations caused by the list of massive
    bodies passed as input.

    The function returns the light-time between source and receiver in TDB
    seconds.

    :param bcrf_rx: RX epoch in TDB
    :param xrec_bcrf_rx: BCRF position vector of receiver at RX epoch
    :param source_id: Name or ID of source known to SPICE
    :param perturbing_bodies: List of names or IDs of massive bodies to be considered for relativistic corrections. Must be known to SPICE
    :param precision: Tolerance to be used in the computation of the light-time
    :param max_iterations: Maximum number of iterations to perform when computing the light-time when requested precision cannot be achieved.
    :return: Light-time between source and receiver in TDB seconds
    """

    # Get BCRF position of source at RX epoch
    et_rx = utils.get_ephemeris_time_from_epoch(tdb_rx)
    xsrc_bcrf_rx: np.ndarray = ephemerides.get_icrf_position_vector(
        source_id, et_rx
    )

    # Check if perturbing bodies are used
    has_bodies: bool = True
    if len(perturbing_bodies) == 0:
        has_bodies = False

    if has_bodies:

        # Get GM and BCRF position of perturbing bodies
        xbodies_bcrf_rx: np.ndarray = np.array(
            [
                ephemerides.get_icrf_position_vector(body_id, et_rx)
                for body_id in perturbing_bodies
            ]
        )
        bodies_gm: np.ndarray = np.array(
            [
                ephemerides.get_body_gravitational_parameter(body_id)
                for body_id in perturbing_bodies
            ]
        )

    # Initialize light-time between transmitter and receiver
    lt_0: np.ndarray = (
        np.linalg.norm(xrec_bcrf_rx - xsrc_bcrf_rx, axis=-1) / CLIGHT
    )

    # Initialize variables for iterative estimation of light-time
    lt_i: np.ndarray = 0.0 * lt_0
    n_i: int = 0

    # Iterative correction of TX epoch
    # Function: F(TX) = RX - TX - R_01/c - RLT_01
    # Derivative: dF/dTX = -1 + (R_01_vec * dR_0_vec/dTX) / (R_01 * c)
    # Newton-Raphson: TX_{i+1} = TX_i - F(TX_i) / dF/dTX
    # Equivalent: LT_{i+1} = LT_i + F(TX_i) / dF/dTX
    while np.any(np.abs(lt_0 - lt_i) > precision) and (n_i < max_iterations):

        # Update light-time and TX epoch
        lt_i = lt_0
        bcrf_tx = tdb_rx - time.TimeDelta(lt_i, format="sec", scale="tdb")

        # Convert TX epoch to ephemeris time
        et_tx = utils.get_ephemeris_time_from_epoch(bcrf_tx)

        # Calculate BCRF coordinates of source at TX epoch
        ssrc_bcrf_tx = ephemerides.get_icrf_state_vector(source_id, et_tx)
        xsrc_bcrf_tx = ssrc_bcrf_tx[:, :3]
        vsrc_bcrf_tx = ssrc_bcrf_tx[:, 3:]

        if has_bodies:
            # Calculate BCRF coordinates of perturbing bodies at TX
            xbodies_bcrf_tx = np.array(
                [
                    ephemerides.get_icrf_position_vector(body, et_tx)
                    for body in perturbing_bodies
                ]
            )

            # Calculate relativistic correction to light-time
            rlt_01: np.ndarray = relativity.post_newtonian_near_field_effect(
                bodies_gm,
                xrec_bcrf_rx,
                xsrc_bcrf_tx,
                xbodies_bcrf_rx,
                xbodies_bcrf_tx,
                consider_bending=True,
            )
        else:
            rlt_01: np.ndarray = np.zeros(len(et_rx))

        # Relative position between receiver and source (non-aberrated)
        r01 = xrec_bcrf_rx - xsrc_bcrf_tx  # (N, 3)
        r01_mag = np.linalg.norm(r01, axis=-1)  # (N,)

        # Evaluate function and derivative for Newton-Raphson
        f = lt_i - (r01_mag / CLIGHT) - rlt_01
        dfdtx = (
            -1.0
            + np.sum((r01 / r01_mag[:, None]) * vsrc_bcrf_tx, axis=-1) / CLIGHT
        )

        # Update light-time
        lt_0 = lt_i + f / dfdtx

        # Update iteration counter
        n_i += 1

    return time.TimeDelta(lt_0, format="sec", scale="tdb")


def light_time_from_tx_epoch(
    tdb_tx: "time.Time",
    xsrc_bcrf_tx: np.ndarray,
    receiver_id: str,
    perturbing_bodies: list[str],
    precision: float,
    max_iterations: int,
) -> "time.TimeDelta":
    """Light-time between source and receiver from TX epoch

    Given a transmission epoch and the associated position of the transmitter in
    BCRF, calculates the light-time for a source (transmitter) identified by a
    name or ID known to SPICE, using the Newton-Raphson algorithm as formulated in Duev 2012 (10.1051/0004-6361/201218885).

    The estimated light-time results from iteratively solving the light-time
    equation with precision settings specified by the user, and considering
    post-Newtonian relativistic perturbations caused by the list of massive
    bodies passed as input.

    The function returns the light-time between source and receiver in TDB
    seconds.

    :param tdb_tx: TX epoch in TDB
    :param xsrc_bcrf_tx: BCRF position vector of source at TX epoch
    :param receiver_id: Name or ID of receiver known to SPICE
    :param perturbing_bodies: List of names or IDs of massive bodies to be considered for relativistic corrections. Must be known to SPICE
    :param precision: Tolerance to be used in the computation of the light-time
    :param max_iterations: Maximum number of iterations to perform when computing the light-time when requested precision cannot be achieved.
    :return: Light-time between source and receiver in TDB seconds
    """

    # Get BCRF position of receiver at TX epoch
    et_tx = utils.get_ephemeris_time_from_epoch(tdb_tx)
    xrec_bcrf_tx: np.ndarray = ephemerides.get_icrf_position_vector(
        receiver_id, et_tx
    )

    # Check if perturbing bodies are used
    has_bodies: bool = True
    if len(perturbing_bodies) == 0:
        has_bodies = False

    if has_bodies:

        # Get GM and BCRF position of perturbing bodies
        xbodies_bcrf_tx: np.ndarray = np.array(
            [
                ephemerides.get_icrf_position_vector(body_id, et_tx)
                for body_id in perturbing_bodies
            ]
        )
        bodies_gm: np.ndarray = np.array(
            [
                ephemerides.get_body_gravitational_parameter(body_id)
                for body_id in perturbing_bodies
            ]
        )

    # Initialize light-time between transmitter and receiver
    lt_0: np.ndarray = (
        np.linalg.norm(xrec_bcrf_tx - xsrc_bcrf_tx, axis=-1) / CLIGHT
    )

    # Initialize variables for iterative estimation of light-time
    lt_i: np.ndarray = 0.0 * lt_0
    n_i: int = 0

    # Iterative correction of TX epoch
    # Function: F(TX) = RX - TX - R_01/c - RLT_01
    # Derivative: dF/dTX = -1 + (R_01_vec * dR_0_vec/dTX) / (R_01 * c)
    # Newton-Raphson: TX_{i+1} = TX_i - F(TX_i) / dF/dTX
    # Equivalent: LT_{i+1} = LT_i + F(TX_i) / dF/dTX
    while np.any(np.abs(lt_0 - lt_i) > precision) and (n_i < max_iterations):

        # Update light-time and RX epoch
        lt_i = lt_0
        bcrf_rx = tdb_tx + time.TimeDelta(lt_i, format="sec", scale="tdb")

        # Convert RX epoch to ephemeris time
        et_rx = utils.get_ephemeris_time_from_epoch(bcrf_rx)

        # Calculate BCRF coordinates of receiver at RX epoch
        srec_bcrf_rx = ephemerides.get_icrf_state_vector(receiver_id, et_rx)
        xrec_bcrf_rx = srec_bcrf_rx[:, :3]
        vrec_bcrf_rx = srec_bcrf_rx[:, 3:]

        if has_bodies:
            # Calculate BCRF coordinates of perturbing bodies at RX
            xbodies_bcrf_rx = np.array(
                [
                    ephemerides.get_icrf_position_vector(body, et_rx)
                    for body in perturbing_bodies
                ]
            )

            # Calculate relativistic correction to light-time
            rlt_01: np.ndarray = relativity.post_newtonian_near_field_effect(
                bodies_gm,
                xrec_bcrf_rx,
                xsrc_bcrf_tx,
                xbodies_bcrf_rx,
                xbodies_bcrf_tx,
                consider_bending=True,
            )
        else:
            rlt_01: np.ndarray = np.zeros(len(et_rx))

        # Relative position between receiver and source (non-aberrated)
        r01 = xrec_bcrf_rx - xsrc_bcrf_tx  # (N, 3)
        r01_mag = np.linalg.norm(r01, axis=-1)  # (N,)

        # Evaluate function and derivative for Newton-Raphson
        f = lt_i - (r01_mag / CLIGHT) - rlt_01
        dfdtx = (
            -1.0
            + np.sum((r01 / r01_mag[:, None]) * vrec_bcrf_rx, axis=-1) / CLIGHT
        )

        # Update light-time
        lt_0 = lt_i + f / dfdtx
        rx_np1 = tdb_tx.tdb + time.TimeDelta(lt_0, format="sec", scale="tdb")

        # Update iteration counter
        n_i += 1

    return time.TimeDelta(lt_0, format="sec", scale="tdb")
