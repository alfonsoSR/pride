"""Transformations between coordinate systems

About EOP bulletins
--------------------

According to the *Explanatory supplement to IERS bulletins A and B* [iers_bulletins]_, the IERS distributes two collections of EOPs: bulletins A and B. Bulletin A is published daily and is only meant to be used while bulletin B is not available. The IERS also distributes long-term collections of EOPs (e.g. C04), which are consistent with bulletin B.

PRIDE accesses EOP information via Astropy, which includes both types of bulletins, and defaults to working with bulletin B unless the user specifies otherwise. As of May of 2025, Astropy's bulletin B corresponds to the C04 20 series, which is consistent with ITRF2020.

.. warning:: As of version 1.2, the station coordinates in the internal catalogs are given in ITRF2000, while the default EOPs are consistent with ITRF2020. PRIDE is under development, and this issue will be fixed in the future.

.. [iers_bulletins] https://hpiers.obspm.fr/iers/bul/bulb/explanatory.pdf

"""

from .core import EOP, seu2itrf, icrf2itrf, itrf2icrf

__all__ = [
    "EOP",
    "seu2itrf",
    "icrf2itrf",
    "itrf2icrf",
]
