from ..core import Doppler
from typing import TYPE_CHECKING, Any
from ...logger import log
from astropy import time, coordinates
from ftplib import FTP_TLS
import unlzw3
import gzip
import numpy as np
from scipy import interpolate
from ... import io, utils

if TYPE_CHECKING:
    from pathlib import Path
    from ...experiment.observation import Observation


TURNAROUND_RATIO = io.load_catalog("config.yaml")["Configuration"]["tr_ratio"]


class Dop(Doppler):
    """Doppler model
    """

    def ensure_resources(self) -> None:

        return None

    def load_resources(self) -> dict[str, Any]:

        return dict()
    
    # Derivative function from original PyPRIDE - TO BE MOVED / REIMPLEMENTED
    def derivative(self, x, y, points=5, poly=2):
        #Calculate a derivative dy/dx at each x
        #Don't forget to normalise output

        dydx = np.zeros_like(y)
        for n0, xi in enumerate(x):
            # number of points to cut from the left-hand side
            nl = int(np.floor(points / 2.0))
            # number of points to cut from the right-hand side
            nr = int(np.ceil(points / 2.0))
            # check/correct bounds:
            if len(x[:n0]) < nl:
                nr = nr + nl - len(x[:n0])
                nl = len(x[:n0])
            if len(x[n0:]) < nr:
                nl = nl + nr - len(x[n0:])
                nr = len(x[n0:])

            # make a fit
            yfit = np.polyfit(x[n0 - nl : n0 + nr], y[n0 - nl : n0 + nr], poly)
            dydx[n0] = np.polyval(np.polyder(yfit), xi)

        return dydx
    
    # Calculate fonafe
    # TO BE IMPLEMENTED
    def calc_fonafe(self, obs: "Observation"):

        fonafe = np.ones_like(obs.tstamps.jd)

        return fonafe

    def doppler_bc(
        tjd, t_1, dd, state_ss_t1, tdb, bcrs, GM,
        TDB_TCB, L_C, C, x_way, freq_type, ut1, sta1,
        sta2=None, L_G=None, AE=None, J_2=None, utc=None,
        gcrs=None, t_utc=None, eops=None, inp=None):
        """
        Doppler calculation following Moyer/Duev
        For reference see Duev PhD thesis, MSU 2012.

        dd - number of days since the start epoch of the ephemeris * 86400
        state_ss_t1  - Solar system bodies r, v (and a for Earth) at t_1 wrt SSBC
        """

        r_1 = sta1.r_GCRS
        v_1 = sta1.v_GCRS

        if sta2 is not None:
            r_2 = sta2.r_GCRS
            v_2 = sta2.v_GCRS
            a_2 = sta2.a_GCRS

        earth = state_ss_t1[2]
        sun = state_ss_t1[-1]

        # Find potential U:
        U = GM[10] / norm(sun[:, 0] - earth[:, 0])

        # BCRS radius vectors of the reception site at t_1:
        R_1 = earth[:, 0] + (1.0 - U / (C**2) - L_C) * r_1 - dot(earth[:, 1], r_1) * earth[:, 1] / (2.0 * C**2)

        V_1 = earth[:, 1] + ((1.0 - U / (C**2) - L_C) * v_1 - dot(earth[:, 1], v_1) * earth[:, 1] / (2.0 * C**2)) * (
            1.0 - (U + v_1**2 / 2.0 - L_C) / C**2)

        """ calculate downleg light-time from S/C to Receiver
            to find signal transmission t_0 time given the reception time t_1
        """
        precision = 1e-16
        n_max = 3
        lag_order = 9

        # initial approximation:
        nn = 0
        lt_01_tmp = 0.0

        # s/c:
        x, _ = lagint(lag_order, tdb, bcrs[:, 6], dd + t_1)
        y, _ = lagint(lag_order, tdb, bcrs[:, 7], dd + t_1)
        z, _ = lagint(lag_order, tdb, bcrs[:, 8], dd + t_1)
        R_0 = np.hstack((x, y, z))

        lt_01 = norm(R_1 - R_0) / C
        t_0 = t_1 - lt_01

        while (abs(lt_01 - lt_01_tmp) > precision) and (nn < n_max):
            lt_01_tmp = lt_01
            t_0 = t_1 - lt_01

            # Coordinates of the spacecraft at t0 in bcrs
            x, _ = lagint(lag_order, tdb, bcrs[:, 6], dd + t_0)
            y, _ = lagint(lag_order, tdb, bcrs[:, 7], dd + t_0)
            z, _ = lagint(lag_order, tdb, bcrs[:, 8], dd + t_0)
            vx, _ = lagint(lag_order, tdb, bcrs[:, 9], dd + t_0)
            vy, _ = lagint(lag_order, tdb, bcrs[:, 10], dd + t_0)
            vz, _ = lagint(lag_order, tdb, bcrs[:, 11], dd + t_0)
            R_0 = np.hstack((x, y, z))
            V_0 = np.hstack((vx, vy, vz))

            # vector needed for RLT calculation
            R_01 = R_1 - R_0

            # >> SS bodies
            RLT = 0.0
            for ii, state in enumerate(state_ss_t1):
                if ii == 2 and norm(r_1) == 0.0:
                    continue
                rb = state[:, 0]
                vb = state[:, 1]
                R_0_B = R_0 - (rb - lt_01 * vb)
                R_1_B = R_1 - rb
                R_01_B = R_1_B - R_0_B
                RLT += (2.0 * GM[ii] / C**3) * log((norm(R_0_B) + norm(R_1_B) + norm(R_01_B) + 2.0 * GM[ii] / C**2) / (norm(R_0_B) + norm(R_1_B) - norm(R_01_B) + 2.0 * GM[ii] / C**2))

            lt_01 = lt_01 - (lt_01 - norm(R_01) / C - RLT) / (1.0 - dot(R_01, V_0) / (C * norm(R_01)))

            t_0 = t_1 - lt_01
            nn += 1

        x, _ = lagint(lag_order, tdb, bcrs[:, 6], dd + t_0)
        y, _ = lagint(lag_order, tdb, bcrs[:, 7], dd + t_0)
        z, _ = lagint(lag_order, tdb, bcrs[:, 8], dd + t_0)
        vx, _ = lagint(lag_order, tdb, bcrs[:, 9], dd + t_0)
        vy, _ = lagint(lag_order, tdb, bcrs[:, 10], dd + t_0)
        vz, _ = lagint(lag_order, tdb, bcrs[:, 11], dd + t_0)
        R_0 = np.hstack((x, y, z))
        V_0 = np.hstack((vx, vy, vz))

        """ BCRS state vectors of celestial bodies at t_0, [m, m/s]: """
        ## Earth:
        JD = tjd
        rrd = pleph(JD + t_0 / 86400.0, 3, 12, inp["jpl_eph"])
        earth = np.vstack(rrd).T * 1e3
        ## Sun:
        rrd = pleph(JD + t_0 / 86400.0, 11, 12, inp["jpl_eph"])
        sun = np.vstack(rrd).T * 1e3
        ## Moon:
        rrd = pleph(JD + t_0 / 86400.0, 10, 12, inp["jpl_eph"])
        moon = np.vstack(rrd).T * 1e3

        state_ss_t0 = []
        for jj in (1, 2, 4, 5, 6, 7, 8, 9):
            rrd = pleph(JD + t_0 / 86400.0, jj, 12, inp["jpl_eph"])
            state_ss_t0.append(np.vstack(rrd).T * 1e3)
        state_ss_t0.insert(2, earth)
        state_ss_t0.append(moon)
        state_ss_t0.append(sun)

        """ My algorithm from PhD thesis """
        # direction vector
        n_b = (R_0 - R_1) / norm(R_0 - R_1)
        # >> calculate dtau_s(t_s)/dTCB, dtau_o(t_s)/dTCB and f_b(t_o)/f_b(t_s)
        GMnaR_s = 0.0
        GMnaR_o = 0.0
        z_sh = 0.0
        for ii, (state_t0, state_t1, gm) in enumerate(zip(state_ss_t0, state_ss_t1, GM)):
            if ii == 2 and norm(r_1) == 0.0:
                continue
            rb_t0 = state_t0[:, 0]
            vb_t0 = state_t0[:, 1]
            rb_t1 = state_t1[:, 0]
            vb_t1 = state_t1[:, 1]
            R_0_B = R_0 - rb_t0
            R_1_B = R_1 - rb_t1
            R_01_B = R_1_B - R_0_B
            V_0_B = V_0 - vb_t0
            V_1_B = V_1 - vb_t1
            GMnaR_s += gm / norm(R_0_B)
            GMnaR_o += gm / norm(R_1_B)
            z_sh += (
                (4.0 * gm / C**3)
                * (
                    (norm(R_1_B) + norm(R_0_B)) * dot(R_01_B, V_1_B - V_0_B) / norm(R_01_B)
                    - norm(R_01_B) * (dot(R_1_B, V_1_B) / norm(R_1_B) + dot(R_0_B, V_0_B) / norm(R_0_B) + 2.0 * gm / C**2)
                )
                / ((norm(R_1_B) + norm(R_0_B) + 2.0 * gm / C**2) ** 2 - norm(R_01_B) ** 2)
            )

        z_sh = -z_sh

        dtau_s_po_dTCB = 1.0 - (GMnaR_s + (norm(V_0) ** 2) / 2.0) / C**2
        dtau_o_po_dTCB = 1.0 - (GMnaR_o + (norm(V_1) ** 2) / 2.0) / C**2

        # >> put everything together
        fonafe = (1.0 + z_sh) * dtau_s_po_dTCB * (1.0 + dot(n_b, V_1) / C) / (dtau_o_po_dTCB * (1.0 + dot(n_b, V_0) / C))

        """ correct fonafe if f is given in GC (i.e. it's not proper): """
        if x_way == "one" and freq_type == "gc":
            #        raise NotImplemented
            x, _ = lagint(lag_order, utc, gcrs[:, 6], dd + t_utc - lt_01)
            y, _ = lagint(lag_order, utc, gcrs[:, 7], dd + t_utc - lt_01)
            z, _ = lagint(lag_order, utc, gcrs[:, 8], dd + t_utc - lt_01)
            vx, _ = lagint(lag_order, utc, gcrs[:, 9], dd + t_utc - lt_01)
            vy, _ = lagint(lag_order, utc, gcrs[:, 10], dd + t_utc - lt_01)
            vz, _ = lagint(lag_order, utc, gcrs[:, 11], dd + t_utc - lt_01)
            r_sc = np.hstack((x, y, z))
            v_sc = np.hstack((vx, vy, vz))

            U_E_sc = GM[2] / norm(r_sc) + GM[2] * AE**2 * J_2 * (1 - 3 * (r_sc[2] / norm(r_sc)) ** 2) / (2.0 * norm(r_sc) ** 3)

            fonafe /= 1 + L_G - ((norm(v_sc) ** 2) / 2.0 + U_E_sc) / C**2
            return fonafe
        
        if x_way == "one":
            return fonafe

        """ 2(3)-way Doppler:
            calculate upleg light-time from Transmitter to S/C
            to find signal transmission time t_2 given the reception time t_0
        """
        # BCRS radius vectors of the transmitting site at t_1:
        R_2_t_1 = earth[:, 0] + (1.0 - U / (C**2) - L_C) * r_2 - dot(earth[:, 1], r_2) * earth[:, 1] / (2.0 * C**2)

        """ calculate upleg light-time from Transmitter to S/C
            to find signal transmission t_2 time given the reception time t_0
        """
        # initial approximation:
        nn = 0
        lt_20_tmp = 0.0

        # initial approximation using R_2_t_1:
        lt_20 = norm(R_0 - R_2_t_1) / C
        t_2 = t_0 - lt_20

        ###############
        JD = tjd
        mjd = tjd - 2400000.5
        astropy_t_2 = Time(
            mjd, t_2 / 86400.0, format="mjd", scale="tdb", precision=9, location=EarthLocation.from_geocentric(*sta2.r_GTRS, unit=units.m)
        )
        # t_2 might be negative. redefine JD and mjd therefore? wozu?
        UTC = astropy_t_2.utc.jd2
        mjd_keep = mjd
        if UTC < 0:
            UTC, JD, mjd = UTC + 1, astropy_t_2.utc.jd1 - 1, astropy_t_2.utc.mjd - 1

        """ compute tai & tt """
        TAI, TT = taitime(mjd, UTC)
        """ interpolate eops to tstamp """
        UT1, eop_int = eop_iers(mjd, UTC, eops)

        """ compute coordinate time fraction of CT day at 2nd observing site """
        CT, dTAIdCT = t_eph(JD, UT1, TT, sta2.lon_gcen, sta2.u, sta2.v)

        """ BCRS state vectors of celestial bodies at JD+CT, [m, m/s]: """
        ## Earth:
        rrd = pleph(JD + CT, 3, 12, inp["jpl_eph"])
        earth = np.vstack(rrd).T * 1e3
        # Earth's acceleration in m/s**2:
        v_plus = np.array(pleph(JD + CT + 1.0 / 86400.0, 3, 12, inp["jpl_eph"])[1])
        v_minus = np.array(pleph(JD + CT - 1.0 / 86400.0, 3, 12, inp["jpl_eph"])[1])
        a = (v_plus - v_minus) * 1e3 / 2.0
        a = np.array(np.matrix(a).T)
        earth = np.hstack((earth, a))
        ## Sun:
        rrd = pleph(JD + CT, 11, 12, inp["jpl_eph"])
        sun = np.vstack(rrd).T * 1e3
        ## Moon:
        rrd = pleph(JD + CT, 10, 12, inp["jpl_eph"])
        moon = np.vstack(rrd).T * 1e3

        state_ss_t2 = []
        for jj in (1, 2, 4, 5, 6, 7, 8, 9):
            rrd = pleph(JD + CT, jj, 12, inp["jpl_eph"])
            state_ss_t2.append(np.vstack(rrd).T * 1e3)
        state_ss_t2.insert(2, earth)
        state_ss_t2.append(moon)
        state_ss_t2.append(sun)

        """ rotation matrix IERS """
        tstamp = astropy_t_2.utc.datetime
        r2000 = ter2cel(tstamp, eop_int, dTAIdCT, "iau2000")

        """ displacements due to geophysical effects """
        if sta2.name == "GEOCENTR":
            pass
        else:
            # displacement due to solid Earth tides:
            sta2 = dehanttideinel(sta2, tstamp, earth, sun, moon, r2000)
            # displacement due to ocean loading:
            sta2 = hardisp(sta2, tstamp, r2000)
            # rotational deformation due to pole tide:
            sta2 = poletide(sta2, tstamp, eop_int, r2000)

        """ add up geophysical corrections and convert sta state to J2000 """
        sta2.j2000gp(r2000)

        r_2 = sta2.r_GCRS
        v_2 = sta2.v_GCRS
        a_2 = sta2.a_GCRS

        # BCRS radius vectors of the transmitting site at t_2_0:
        R_2_t_2 = earth[:, 0] + (1.0 - U / (C**2) - L_C) * r_2 - dot(earth[:, 1], r_2) * earth[:, 1] / (2.0 * C**2)
        V_2_t_2 = (
            earth[:, 1]
            + (1.0 - 2.0 * U / C**2 - 0.5 * (norm(earth[:, 1]) / C) ** 2 - dot(earth[:, 1], v_2) / C**2) * v_2
            - 0.5 * dot(earth[:, 1], v_2) * earth[:, 1] / C**2
        )
        A_2_t_2 = (
            earth[:, 2]
            + (1.0 - 3.0 * U / C**2 - (norm(earth[:, 1]) / C) ** 2 + L_C - 2.0 * dot(earth[:, 1], v_2) / C**2) * a_2
            - 0.5 * dot(earth[:, 1], a_2) * (earth[:, 1] + 2.0 * v_2) / C**2
        )

        t_2_0 = deepcopy(t_2)
        ##############

        while (abs(lt_20 - lt_20_tmp) > precision) and (nn < n_max):
            lt_20_tmp = deepcopy(lt_20)
            t_2 = t_0 - lt_20

            R_2 = R_2_t_2 + V_2_t_2 * (t_2 - t_2_0) + 0.5 * A_2_t_2 * (t_2 - t_2_0) ** 2
            V_2 = V_2_t_2 + A_2_t_2 * (t_2 - t_2_0)

            # vector needed for RLT calculation
            R_20 = R_0 - R_2

            # >> SS bodies
            RLT = 0.0
            for ii, (state_t0, state_t2) in enumerate(zip(state_ss_t0, state_ss_t2)):
                if ii == 2 and norm(r_2) == 0.0:
                    continue
                rb_t0 = state_t0[:, 0]
                #            vb_t0 = state_t0[:,1]
                rb_t2 = state_t2[:, 0]
                vb_t2 = state_t2[:, 1]
                R_0_B = R_0 - rb_t0
                R_2_B = R_2 - (rb_t2 + (t_2 - t_2_0) * vb_t2)
                R_20_B = R_0_B - R_2_B
                RLT += (2.0 * GM[ii] / C**3) * log((norm(R_2_B) + norm(R_0_B) + norm(R_20_B) + 2.0 * GM[ii] / C**2) / (norm(R_2_B) + norm(R_0_B) - norm(R_20_B) + 2.0 * GM[ii] / C**2))

            lt_20 = lt_20 - (lt_20 - norm(R_20) / C - RLT) / (1.0 - dot(R_20, V_2) / (C * norm(R_20)))

            t_2 = t_0 - lt_20
            nn += 1

        # transmitter state at found t_2
        R_2 = R_2_t_2 + V_2_t_2 * (t_2 - t_2_0) + 0.5 * A_2_t_2 * (t_2 - t_2_0) ** 2
        V_2 = V_2_t_2 + A_2_t_2 * (t_2 - t_2_0)

        # unfix mjd if necessary:
        mjd = mjd_keep

        astropy_t_2 = Time(mjd, t_2 / 86400.0, format="mjd", scale="tdb", precision=9, location=EarthLocation.from_geocentric(*sta2.r_GTRS, unit=units.m))
        t_2_UTC = astropy_t_2.utc.jd2

        """ My algorithm from PhD thesis """
        # direction vector
        n_b = (R_2 - R_0) / norm(R_2 - R_0)
        # >> calculate dtau_s(t_s)/dTCB, dtau_o(t_s)/dTCB and f_b(t_o)/f_b(t_s)
        GMnaR_s = 0.0
        GMnaR_o = 0.0
        z_sh = 0.0
        for ii, (state_t0, state_t2, gm) in enumerate(zip(state_ss_t0, state_ss_t2, GM)):
            if ii == 2 and norm(r_2) == 0.0:
                continue
            rb_t0 = state_t0[:, 0]
            vb_t0 = state_t0[:, 1]
            rb_t2 = state_t2[:, 0]
            vb_t2 = state_t2[:, 1]
            R_0_B = R_0 - rb_t0
            R_2_B = R_2 - (rb_t2 + (t_2 - t_2_0) * vb_t2)
            R_20_B = R_0_B - R_2_B
            V_0_B = V_0 - vb_t0
            V_2_B = V_2 - vb_t2
            GMnaR_s += gm / norm(R_2_B)
            GMnaR_o += gm / norm(R_0_B)
            z_sh += (
                (4.0 * gm / C**3)
                * (
                    (norm(R_0_B) + norm(R_2_B)) * dot(R_20_B, V_0_B - V_2_B) / norm(R_20_B)
                    - norm(R_20_B) * (dot(R_0_B, V_0_B) / norm(R_0_B) + dot(R_2_B, V_2_B) / norm(R_2_B) + 2.0 * gm / C**2)
                )
                / ((norm(R_0_B) + norm(R_2_B) + 2.0 * gm / C**2) ** 2 - norm(R_20_B) ** 2)
            )

        z_sh = -z_sh

        dtau_s_po_dTCB = 1.0 - (GMnaR_s + (norm(V_2) ** 2) / 2.0) / C**2
        dtau_o_po_dTCB = 1.0 - (GMnaR_o + (norm(V_0) ** 2) / 2.0) / C**2

        # >> put everything together
        fonafe *= (1.0 + z_sh) * dtau_s_po_dTCB * (1.0 + dot(n_b, V_0) / C) / (dtau_o_po_dTCB * (1.0 + dot(n_b, V_2) / C))

        # correction if station 1 is GC:
        if sta1.name == "GEOCENTR":
            fonafe *= 1 + L_G

        return fonafe


    def calculate(self, obs: "Observation") -> Any:

        fonafe = self.calc_fonafe(obs)

        # Get frequency of the detected signal
        freq = np.zeros_like(obs.tstamps.jd)
        if obs.source.is_farfield:
            freq += obs.band.channels[0].sky_freq

        if obs.source.is_nearfield:

            # Get uplink and downlink TX epochs in TDB
            light_time = obs.tstamps.tdb - obs.tx_epochs.tdb  # type: ignore
            uplink_tx = obs.tx_epochs.tdb - light_time  # type: ignore
            downlink_tx = obs.tx_epochs.tdb  # type: ignore

            # Read three-way ramping data
            if obs.source.has_three_way_ramping:

                three_way = obs.source.three_way_ramping
                mask_3way = (
                    uplink_tx.jd[:, None] >= three_way["t0"].jd[None, :]  # type: ignore
                ) * (
                    uplink_tx.jd[:, None] <= three_way["t1"].jd[None, :]  # type: ignore
                )
                f0 = np.sum(np.where(mask_3way, three_way["f0"], 0), axis=1)
                df0 = np.sum(np.where(mask_3way, three_way["df"], 0), axis=1)
                t0 = np.sum(np.where(mask_3way, three_way["t0"].jd, 0), axis=1)
                dt = time.TimeDelta(uplink_tx.jd - t0, format="jd").to("s").value  # type: ignore
                freq += (f0 + df0 * dt) * TURNAROUND_RATIO

                # Check for lack of coverage
                holes = np.sum(mask_3way, axis=1) == 0
            else:
                holes = np.ones_like(obs.tstamps.jd, dtype=int)

            # Fill holes in three-way ramping with one-way data
            if np.any(holes) and obs.source.has_one_way_ramping:

                one_way = obs.source.one_way_ramping
                mask_1way = (
                    (downlink_tx.jd[:, None] >= one_way["t0"].jd[None, :])  # type: ignore
                    * (downlink_tx.jd[:, None] <= one_way["t1"].jd[None, :])  # type: ignore
                    * holes[:, None]
                )
                f0 = np.sum(np.where(mask_1way, one_way["f0"], 0), axis=1)
                df0 = np.sum(np.where(mask_1way, one_way["df"], 0), axis=1)
                t0 = np.sum(np.where(mask_1way, one_way["t0"].jd, 0), axis=1)
                dt = (
                    time.TimeDelta(downlink_tx.jd - t0, format="jd")  # type: ignore
                    .to("s")
                    .value
                )
                freq += f0 + df0 * dt

                # Re-check for lack of coverage
                holes *= np.sum(mask_1way, axis=1) == 0

            # Fill remaining holes with constant frequency
            freq += np.where(holes, obs.source.default_frequency, 0)

        doppler_raw = freq * fonafe
        doppler = doppler_raw

        tp = obs.tstamps * 86400.00

        drate = self.derivative(tp, obs.delays, points=5, poly=2)

        doppler = np.vstack((doppler, -drate * doppler_raw)).T

        return doppler
