from ..core import Delay
from ... import io
from typing import TYPE_CHECKING, Any
from ...logger import log
from astropy import time
import numpy as np

import spiceypy as spice
from ...source import FarFieldSource, NearFieldSource
from ...constants import J2000, L_C

if TYPE_CHECKING:
    from ...experiment.observation import Observation


class Geometric(Delay):
    """Geometric delay"""

    name = "Geometric"
    etc = {}
    requires_spice = True
    station_specific = False

    def ensure_resources(self) -> None:

        # NOTE: This has been moved to the constructor of the Experiment class
        # # Initialize SPICE kernels manager
        # kernel_manager = io.SpiceKernelManager(
        #     mission=self.exp.setup.general["target"],
        #     kernels_folder=self.config["data"],
        # )

        # # Download metakernel if not present
        # metakernel = kernel_manager.ensure_metakernel()

        # # Download SPICE kernels in the metakernel
        # kernel_manager.ensure_kernels(metakernel)

        return None

    def load_resources(self) -> dict[str, Any]:

        log.debug(f"Loading resources for {self.name} delay")

        return {}

    def calculate_nearfield(self, obs: "Observation") -> np.ndarray:
        """Calculate geometric delay for a near-field source

        NOTE: This function assumes that the phase center is the geocenter.
        """

        # log.warning(
        #     "Implementation of near-field geometric delay is not reliable"
        # )

        # Sanity
        source = obs.source
        assert isinstance(source, NearFieldSource)

        # Get TX epoch at spacecraft [Downlink SC -> Station]
        tx = obs.tx_epochs
        rx_station: time.Time = obs.tstamps.tdb  # type: ignore
        assert tx.scale == "tdb"  # Sanity
        assert rx_station.scale == "tdb"  # Sanity
        lt_station: np.ndarray = (rx_station - tx).to("s").value  # type: ignore

        # Calculate RX epoch at phase center [Geocenter]
        #######################################################################

        # Initialization
        clight = spice.clight() * 1e3
        clight2 = clight * clight

        # Calculate BCRF position of source at TX epoch
        et_tx: np.ndarray = (tx - J2000.tdb).to("s").value  # type: ignore
        xsrc_bcrf_tx = (
            np.array(
                spice.spkpos(source.spice_id, et_tx, "J2000", "NONE", "SSB")[0]
            )
            * 1e3
        )

        # Calculate BCRF position of phase center at estimated RX epoch
        # NOTE: Seems logical to initialize RX with the one of the station
        et_rx: np.ndarray = (rx_station - J2000.tdb).to("s").value  # type: ignore
        xphc_bcrf_rx = (
            np.array(spice.spkpos("EARTH", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )

        # Calculate gravitational parameter of celestial bodies
        _bodies: list = self.exp.setup.internal["lt_correction_bodies"]
        bodies = [bi for bi in _bodies if bi != "earth"]
        bodies_gm = (
            np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
            * 1e9
        )

        # Calculate BCRF positions of celestial bodies at TX
        xbodies_bcrf_tx = np.array(
            [
                np.array(spice.spkpos(body, et_tx, "J2000", "NONE", "SSB")[0])
                * 1e3
                for body in bodies
            ]
        )

        # Calculate position of source wrt celestial bodies at TX
        r0b = xsrc_bcrf_tx[None, :, :] - xbodies_bcrf_tx
        r0b_mag = np.linalg.norm(r0b, axis=-1)  # (M, N)

        # Initialize light travel time and rx epoch
        # lt_np1 = LT_{n+1}
        lt_np1 = np.linalg.norm(xphc_bcrf_rx - xsrc_bcrf_tx, axis=-1) / clight
        rx_np1: time.Time = tx + time.TimeDelta(lt_np1, format="sec")

        # Initialize variables for iteration
        lt_n = 0.0 * lt_np1
        n_iter = 0
        iter_max = self.exp.setup.internal["lt_max_iterations"]
        precision = float(self.exp.setup.internal["lt_precision"])

        # Iterative correction of LT and RX
        # Function: F(RX) = RX - TX - R_02/c - RLT_02
        # Derivative: dF/dRX = 1 - (R_02_vec * dR_2_vec/dRX) / (R_02 * c)
        # Newton-Raphson: RX_{n+1} = RX_n - F(RX_n) / dF/dRX
        # Equivalent: LT_{n+1} = LT_n - F(RX_n) / dF/dRX
        while np.any(np.abs(lt_np1 - lt_n) > precision) and n_iter < iter_max:

            # Update light travel time and RX
            lt_n = lt_np1
            rx_n = tx + time.TimeDelta(lt_n, format="sec")

            # Convert RX to ephemeris time
            et_rx = (rx_n - J2000.tdb).to("s").value  # type: ignore

            # Calculate BCRF coordinates of phase center at RX
            sphc_bcrf_rx = (
                np.array(
                    spice.spkezr("EARTH", et_rx, "J2000", "NONE", "SSB")[0]
                )
                * 1e3
            )
            xphc_bcrf_rx = sphc_bcrf_rx[:, :3]
            vphc_bcrf_rx = sphc_bcrf_rx[:, 3:]

            # Calculate BCRF positions of celestial bodies at RX
            xbodies_bcrf_rx = np.array(
                [
                    np.array(
                        spice.spkpos(body, et_rx, "J2000", "NONE", "SSB")[0]
                    )
                    * 1e3
                    for body in bodies
                ]
            )

            # Calculate relativistic correction
            r02 = xphc_bcrf_rx - xsrc_bcrf_tx  # (N, 3)
            r02_mag = np.linalg.norm(r02, axis=-1)  # (N,)
            r2b = xphc_bcrf_rx[None, :, :] - xbodies_bcrf_rx  # (M, N, 3)
            r2b_mag = np.linalg.norm(r2b, axis=-1)  # (M, N)
            r02b_mag = np.linalg.norm(r2b - r0b, axis=-1)  # (M, N)
            gmc = 2.0 * bodies_gm[:, None] / clight2  # (M, 1)
            rlt_02 = np.sum(
                (gmc / clight)
                * np.log(
                    (r0b_mag + r2b_mag + r02b_mag + gmc)
                    / (r0b_mag + r2b_mag - r02b_mag + gmc)
                ),
                axis=0,
            )

            # Evaluate function and derivative
            f = lt_n - (r02_mag / clight) - rlt_02
            dfdrx = (
                1.0
                - np.sum((r02 / r02_mag[:, None]) * vphc_bcrf_rx, axis=-1)
                / clight
            )

            # Update light travel time and RX
            lt_np1 = lt_n - f / dfdrx
            rx_np1 = tx + time.TimeDelta(lt_np1, format="sec")

            # Update iteration
            n_iter += 1

        # Calculate post-newtonian correction for path between stations
        # #####################################################################
        # NOTE: From now on, I refer to the RX of the station as rx1 and to the RX of the phase center as rx2

        # Calculate BCRF position of station at RX1
        xsta_gcrf_rx1 = obs.station.location(obs.tstamps, frame="icrf")
        rx1 = rx_station
        et_rx1 = (rx1 - J2000.tdb).to("s").value  # type: ignore
        searth_bcrf_rx1 = (
            np.array(spice.spkezr("EARTH", et_rx1, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        xearth_bcrf_rx1 = searth_bcrf_rx1[:, :3]
        vearth_bcrf_rx1 = searth_bcrf_rx1[:, 3:]
        xsun_bcrf_rx1 = (
            np.array(spice.spkpos("SUN", et_rx1, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        U_earth = (
            spice.bodvrd("SUN", "GM", 1)[1][0]
            * 1e9
            / np.linalg.norm(xsun_bcrf_rx1 - xearth_bcrf_rx1, axis=-1)
        )
        xsta_bcrf_rx1 = (
            xearth_bcrf_rx1
            + (1.0 - U_earth[:, None] / clight2 - L_C) * xsta_gcrf_rx1
            - 0.5
            * np.sum(vearth_bcrf_rx1 * xsta_gcrf_rx1, axis=-1)[:, None]
            * vearth_bcrf_rx1
            / clight2
        )

        # Calculate BCRF position of phase center at RX2
        rx2 = rx_np1
        et_rx2 = (rx2 - J2000.tdb).to("s").value  # type: ignore
        xphc_bcrf_rx2 = (
            np.array(spice.spkpos("EARTH", et_rx2, "J2000", "NONE", "SSB")[0])
            * 1e3
        )

        # Calculate position of celestial bodies at RX1 and RX2
        xbodies_bcrf_rx1 = (
            np.array(
                [
                    np.array(
                        spice.spkpos(body, et_rx1, "J2000", "NONE", "SSB")[0]
                    )
                    for body in bodies
                ]
            )
            * 1e3
        )
        xbodies_bcrf_rx2 = (
            np.array(
                [
                    np.array(
                        spice.spkpos(body, et_rx2, "J2000", "NONE", "SSB")[0]
                    )
                    for body in bodies
                ]
            )
            * 1e3
        )

        # Calculate relativistic correction
        r01 = xsta_bcrf_rx1 - xsrc_bcrf_tx  # (N, 3)
        r01_mag = np.linalg.norm(r01, axis=-1)  # (N,)
        r02 = xphc_bcrf_rx2 - xsrc_bcrf_tx  # (N, 3)
        r02_mag = np.linalg.norm(r02, axis=-1)  # (N,)
        r0b = xsrc_bcrf_tx[None, :, :] - xbodies_bcrf_tx  # (M, N, 3)
        r0b_mag = np.linalg.norm(r0b, axis=-1)
        r1b = xsta_bcrf_rx1[None, :, :] - xbodies_bcrf_rx1  # (M, N, 3)
        r1b_mag = np.linalg.norm(r1b, axis=-1)
        r2b = xphc_bcrf_rx2[None, :, :] - xbodies_bcrf_rx2  # (M, N, 3)
        r2b_mag = np.linalg.norm(r2b, axis=-1)
        gmc = 2.0 * bodies_gm[:, None] / (clight * clight)  # (M, 1)
        tg_12 = np.sum(
            (gmc / clight)
            * np.log(
                (r2b_mag + r0b_mag + r02_mag)
                * (r1b_mag + r0b_mag - r01_mag)
                / (
                    (r2b_mag + r0b_mag - r02_mag)
                    * (r1b_mag + r0b_mag + r01_mag)
                )
            ),
            axis=0,
        )

        # Calculate delay in TT
        #######################################################################
        dt: np.ndarray = (rx2 - rx1).to("s").value  # type: ignore
        vearth_mag = np.linalg.norm(vearth_bcrf_rx1, axis=-1)
        baseline = -xsta_gcrf_rx1
        v2 = 0.0 * vearth_bcrf_rx1  # Velocity of phase center in GCRF
        return -(
            (dt + tg_12)
            * (1 - (0.5 * vearth_mag * vearth_mag + U_earth) / clight2)
            / (1.0 - L_C)
            - np.sum(vearth_bcrf_rx1 * baseline, axis=-1) / clight2
        ) / (1.0 + np.sum(vearth_bcrf_rx1 * v2, axis=-1) / clight2)

    def calculate_farfield(self, obs: "Observation") -> np.ndarray:
        """Calculate geometric delay for a far-field source

        Calculates the geometric delay using the consensus model for far-field sources, described in section 11 of the IERS Conventions 2010.
        """

        # Sanity
        source = obs.source
        assert isinstance(source, FarFieldSource)

        # Initialization
        clight = spice.clight() * 1e3
        clight2 = clight * clight

        # Calculate baseline vector [For geocenter phase center it is just
        # the GCRF position of the station at RX]
        baseline = obs.station.location(obs.tstamps, frame="icrf")
        xsta_gcrf_rx = baseline

        # Calculate potential at geocenter
        et_rx: np.ndarray = (obs.tstamps.tdb - J2000.tdb).to("s").value  # type: ignore
        searth_bcrf_rx = (
            np.array(spice.spkezr("EARTH", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        xearth_bcrf_rx = searth_bcrf_rx[:, :3]
        vearth_bcrf_rx = searth_bcrf_rx[:, 3:]
        xsun_bcrf_rx = (
            np.array(spice.spkpos("SUN", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        gm_sun = spice.bodvrd("SUN", "GM", 1)[1][0] * 1e9
        U_earth = gm_sun / np.linalg.norm(
            xsun_bcrf_rx - xearth_bcrf_rx, axis=-1
        )

        # Calculate BCRF position of station at RX
        xsta_bcrf_rx = (
            xearth_bcrf_rx
            + (1.0 - U_earth[:, None] / clight2 - L_C) * xsta_gcrf_rx
            - 0.5
            * np.sum(vearth_bcrf_rx * xsta_gcrf_rx, axis=-1)[:, None]
            * vearth_bcrf_rx
            / clight2
        )

        # Position of phase center is just that of Earth for geocenter
        xphc_bcrf_rx = xearth_bcrf_rx

        # Calculate gravitational correction using IERS algorithm
        #######################################################################
        _bodies = self.exp.setup.internal["lt_correction_bodies"]
        bodies = [bi for bi in _bodies if bi != "earth"]
        bodies_gm = (
            np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
            * 1e9
        )
        xbodies_bcrf_rx = np.array(
            [
                np.array(spice.spkpos(body, et_rx, "J2000", "NONE", "SSB")[0])
                * 1e3
                for body in bodies
            ]
        )
        xbodies_sta_bcrf_rx = xbodies_bcrf_rx - xsta_bcrf_rx

        # Estimate time of closest approach to planets
        ks = obs.source.observed_ks  # Pointing vector
        et_planets = (
            et_rx
            - np.sum(ks[None, None, :] * xbodies_sta_bcrf_rx, axis=-1) / clight
        )
        et_closest = np.where(et_rx[None, :] < et_planets, et_rx, et_planets)

        # Calculate position of celestial bodies at closest approach
        xbodies_bcrf_closest = np.array(
            [
                np.array(spice.spkpos(body, et_body, "J2000", "NONE", "SSB")[0])
                * 1e3
                for body, et_body in zip(bodies, et_closest)
            ]
        )

        # Calculate correction
        r1j = xsta_bcrf_rx[None, :, :] - xbodies_bcrf_closest
        r1j_mag = np.linalg.norm(r1j, axis=-1)
        k_r1j = np.sum(ks[None, None, :] * r1j, axis=-1)
        r2j = (
            xphc_bcrf_rx[None, :, :]
            - np.sum(ks[None, :] * baseline, axis=-1)[None, :, None]
            * vearth_bcrf_rx[None, :, :]
            / clight
            - xbodies_bcrf_closest
        )
        r2j_mag = np.linalg.norm(r2j, axis=-1)
        k_r2j = np.sum(ks[None, :] * r2j, axis=-1)
        gmc = 2.0 * bodies_gm[:, None] / (clight * clight2)
        T_g = np.sum(
            gmc * np.log((r1j_mag + k_r1j) / (r2j_mag + k_r2j)), axis=0
        )

        # Calculate delay in TT
        ks_b = np.sum(ks[None, :] * baseline, axis=-1)
        ks_vearth = np.sum(ks[None, :] * vearth_bcrf_rx, axis=-1)
        vearth_b = np.sum(vearth_bcrf_rx * baseline, axis=-1)
        vearth_mag2 = np.linalg.norm(vearth_bcrf_rx, axis=-1) ** 2
        return (
            T_g
            - (ks_b / clight)
            * (1.0 - (2.0 * U_earth / clight2) - (0.5 * vearth_mag2 / clight2))
            - (vearth_b / clight2) * (1.0 + 0.5 * ks_vearth / clight)
        ) / (1.0 + ks_vearth / clight)

    def calculate(self, obs: "Observation") -> Any:

        if isinstance(obs.source, FarFieldSource):
            return self.calculate_farfield(obs)
        elif isinstance(obs.source, NearFieldSource):
            return self.calculate_nearfield(obs)
        else:
            log.error(
                "Failed to calculate geometric delay: Invalid source type"
            )
            exit(1)

        raise NotImplementedError("Missing calculate for geometric")
