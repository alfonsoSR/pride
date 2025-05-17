Requirements on VEX files
==========================

.. warning:: This document is a work in progress!

In order to be processed with PRIDE, a VEX file must fulfill the following requirements

* It must contain a ``$CLOCK`` section, specifying clock parameters for all the antennas involved in the experiment.
* Each source must have an attribute ``source_type``, which must be set to either ``"target"`` or ``"calibrator"``.
