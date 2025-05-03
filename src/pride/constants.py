from astropy import time

J2000: time.Time = time.Time("2000-01-01T12:00:00", scale="tt").utc  # type: ignore
# L_C = 1 - d(TT)/d(TCG) [From IERS Conventions 2010]
L_C: float = 1.48082686741e-8
