from ..core import Delay
from ... import astro, utils
from typing import TYPE_CHECKING, Any
from ...logger import log
from astropy import time
import numpy as np

from ...source import FarFieldSource, NearFieldSource
from ...constants import L_C, CLIGHT

if TYPE_CHECKING:
    from ...experiment.observation import Observation


class Geometric(Delay):
    """Geometric delay"""

    name = "Geometric"

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
        clight2 = CLIGHT * CLIGHT

        # Calculate BCRF position of source at TX epoch
        et_tx = utils.get_ephemeris_time_from_epoch(tx)
        xsrc_bcrf_tx = astro.get_icrf_position_vector(source.spice_id, et_tx)

        # Calculate BCRF position of phase center at estimated RX epoch
        # NOTE: Seems logical to initialize RX with the one of the station
        et_rx = utils.get_ephemeris_time_from_epoch(rx_station)
        xphc_bcrf_rx = astro.get_icrf_position_vector("earth", et_rx)

        # Calculate gravitational parameter of celestial bodies
        _bodies: list = self.exp.setup.internal["lt_correction_bodies"]
        bodies = [bi for bi in _bodies if bi != "earth"]
        bodies_gm = np.array(
            [astro.get_body_gravitational_parameter(body) for body in bodies]
        )

        # Calculate BCRF positions of celestial bodies at TX
        xbodies_bcrf_tx = np.array(
            [astro.get_icrf_position_vector(body, et_tx) for body in bodies]
        )

        # Calculate position of source wrt celestial bodies at TX
        r0b = xsrc_bcrf_tx[None, :, :] - xbodies_bcrf_tx
        r0b_mag = np.linalg.norm(r0b, axis=-1)  # (M, N)

        # Initialize light travel time and rx epoch
        # lt_np1 = LT_{n+1}
        lt_np1 = np.linalg.norm(xphc_bcrf_rx - xsrc_bcrf_tx, axis=-1) / CLIGHT
        rx_np1 = tx + time.TimeDelta(lt_np1, format="sec", scale="tdb")

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
            rx_n = tx + time.TimeDelta(lt_n, format="sec", scale="tdb")

            # Convert RX to ephemeris time
            et_rx = utils.get_ephemeris_time_from_epoch(rx_n)

            # Calculate BCRF coordinates of phase center at RX
            sphc_bcrf_rx = astro.get_icrf_state_vector("earth", et_rx)
            xphc_bcrf_rx = sphc_bcrf_rx[:, :3]
            vphc_bcrf_rx = sphc_bcrf_rx[:, 3:]

            # Calculate BCRF positions of celestial bodies at RX
            xbodies_bcrf_rx = np.array(
                [astro.get_icrf_position_vector(body, et_rx) for body in bodies]
            )

            # Calculate relativistic correction
            r02 = xphc_bcrf_rx - xsrc_bcrf_tx  # (N, 3)
            r02_mag = np.linalg.norm(r02, axis=-1)  # (N,)
            r2b = xphc_bcrf_rx[None, :, :] - xbodies_bcrf_rx  # (M, N, 3)
            r2b_mag = np.linalg.norm(r2b, axis=-1)  # (M, N)
            r02b_mag = np.linalg.norm(r2b - r0b, axis=-1)  # (M, N)
            gmc = 2.0 * bodies_gm[:, None] / clight2  # (M, 1)
            rlt_02 = np.sum(
                (gmc / CLIGHT)
                * np.log(
                    (r0b_mag + r2b_mag + r02b_mag + gmc)
                    / (r0b_mag + r2b_mag - r02b_mag + gmc)
                ),
                axis=0,
            )

            # Evaluate function and derivative
            f = lt_n - (r02_mag / CLIGHT) - rlt_02
            dfdrx = (
                1.0
                - np.sum((r02 / r02_mag[:, None]) * vphc_bcrf_rx, axis=-1)
                / CLIGHT
            )

            # Update light travel time and RX
            lt_np1 = lt_n - f / dfdrx
            rx_np1 = tx + time.TimeDelta(lt_np1, format="sec", scale="tdb")

            # Update iteration
            n_iter += 1

        # Calculate post-newtonian correction for path between stations
        # #####################################################################
        # NOTE: From now on, I refer to the RX of the station as rx1 and to the RX of the phase center as rx2

        # Calculate BCRF position of station at RX1
        xsta_gcrf_rx1 = obs.station.location(obs.tstamps, frame="icrf")
        rx1 = rx_station
        et_rx1 = utils.get_ephemeris_time_from_epoch(rx1)
        searth_bcrf_rx1 = astro.get_icrf_state_vector("earth", et_rx1)
        xearth_bcrf_rx1 = searth_bcrf_rx1[:, :3]
        vearth_bcrf_rx1 = searth_bcrf_rx1[:, 3:]
        xsun_bcrf_rx1 = astro.get_icrf_position_vector("sun", et_rx1)
        U_earth = astro.get_body_gravitational_parameter(
            "sun"
        ) / np.linalg.norm(xsun_bcrf_rx1 - xearth_bcrf_rx1, axis=-1)
        xsta_bcrf_rx1 = astro.transform_position_from_gcrf_to_bcrf(
            xsta_gcrf_rx1, xearth_bcrf_rx1, vearth_bcrf_rx1, U_earth
        )

        # Calculate BCRF position of phase center at RX2
        rx2 = rx_np1
        et_rx2 = utils.get_ephemeris_time_from_epoch(rx2)
        xphc_bcrf_rx2 = astro.get_icrf_position_vector("earth", et_rx2)

        # Calculate position of celestial bodies at RX1 and RX2
        xbodies_bcrf_rx1 = np.array(
            [astro.get_icrf_position_vector(body, et_rx1) for body in bodies]
        )
        xbodies_bcrf_rx2 = np.array(
            [astro.get_icrf_position_vector(body, et_rx2) for body in bodies]
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
        gmc = 2.0 * bodies_gm[:, None] / clight2  # (M, 1)
        tg_12 = np.sum(
            (gmc / CLIGHT)
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
        clight2 = CLIGHT * CLIGHT

        # Calculate baseline vector [For geocenter phase center it is just
        # the GCRF position of the station at RX]
        baseline = obs.station.location(obs.tstamps, frame="icrf")
        xsta_gcrf_rx = baseline

        # Calculate potential at geocenter
        et_rx = utils.get_ephemeris_time_from_epoch(obs.tstamps)
        searth_bcrf_rx = astro.get_icrf_state_vector("earth", et_rx)
        xearth_bcrf_rx = searth_bcrf_rx[:, :3]
        vearth_bcrf_rx = searth_bcrf_rx[:, 3:]
        xsun_bcrf_rx = astro.get_icrf_position_vector("sun", et_rx)
        gm_sun = astro.get_body_gravitational_parameter("sun")
        U_earth = gm_sun / np.linalg.norm(
            xsun_bcrf_rx - xearth_bcrf_rx, axis=-1
        )

        # Calculate BCRF position of station at RX
        xsta_bcrf_rx = astro.transform_position_from_gcrf_to_bcrf(
            xsta_gcrf_rx, xearth_bcrf_rx, vearth_bcrf_rx, U_earth
        )

        # Position of phase center is just that of Earth for geocenter
        xphc_bcrf_rx = xearth_bcrf_rx

        # Calculate gravitational correction using IERS algorithm
        #######################################################################
        _bodies = self.exp.setup.internal["lt_correction_bodies"]
        bodies = [bi for bi in _bodies if bi != "earth"]
        bodies_gm = np.array(
            [astro.get_body_gravitational_parameter(body) for body in bodies]
        )
        xbodies_bcrf_rx = np.array(
            [astro.get_icrf_position_vector(body, et_rx) for body in bodies]
        )
        xbodies_sta_bcrf_rx = xbodies_bcrf_rx - xsta_bcrf_rx

        # Estimate time of closest approach to planets
        ks = obs.source.observed_ks  # Pointing vector
        et_planets = (
            et_rx
            - np.sum(ks[None, None, :] * xbodies_sta_bcrf_rx, axis=-1) / CLIGHT
        )
        et_closest = np.where(et_rx[None, :] < et_planets, et_rx, et_planets)

        # Calculate position of celestial bodies at closest approach
        xbodies_bcrf_closest = np.array(
            [
                astro.get_icrf_position_vector(body, et_body)
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
            / CLIGHT
            - xbodies_bcrf_closest
        )
        r2j_mag = np.linalg.norm(r2j, axis=-1)
        k_r2j = np.sum(ks[None, :] * r2j, axis=-1)
        gmc = 2.0 * bodies_gm[:, None] / (CLIGHT * clight2)
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
            - (ks_b / CLIGHT)
            * (1.0 - (2.0 * U_earth / clight2) - (0.5 * vearth_mag2 / clight2))
            - (vearth_b / clight2) * (1.0 + 0.5 * ks_vearth / CLIGHT)
        ) / (1.0 + ks_vearth / CLIGHT)

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
