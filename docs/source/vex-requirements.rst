Requirements on VEX files
==========================

.. warning:: This document is a work in progress!

In order to be compatible with PRIDE, a VEX file must comply with some additional requirements. The purpose of this page is to provide an overview of these requirements, and describe the expected behavior of the program when they are not met.


.. rubric:: VEX files must comply with the following requirements:

1. Each source must have its ``source_type`` attribute set to ``"target"`` or ``"calibrator"``.
2. The VEX file must contain a ``$CLOCK`` section.
3. Consecutive scans should be separated by, at least, one second.

Each of these requirements is described with more detail in its corresponding section:

.. contents::
    :local:


Source type specification [R1]
------------------------------

When designing an experiment to observe a spacecraft, it is common to discretize its trajectory into a series of points, and treat each of them as an independent source. Although each of these sources is usually given a different name, the program should understand that they refer to the same object. Building upon the assumption that each VEX file can only contain observations of one near-field source, PRIDE relies on the ``source_type`` attribute of the sources to do this.

During the first steps of the processing, PRIDE parses the VEX file to identify all the sources involved in the experiment, and creates a data structure for each of them. Each source of ``calibrator`` type is assigned its own data structure, identified by the source name in the VEX file. However, all the sources of ``target`` type are grouped under a single data structure, identified by the target name the user specified in the configuration file. When the ``source_type`` attribute is not specified in the VEX file, or it takes an unexpected value, the program raises an error, and terminates the execution.

Presence of $CLOCK section [R2]
--------------------------------

To be added in a future version

Separation between consecutive scans [R3]
------------------------------------------

The ``$SCAN`` section of a VEX file contains a sequence of scans for the experiment. Each scan contains metadata about its reference epoch, its target, and operation mode; as well as collections of recording data for each of the involved stations.

.. code-block::
    :caption: Example of scan defintion in VEX file

    scan No0028;
        start=2023y292d15h35m00s; mode=sess323.X256R; source=J1720-2258;
        station=Wb:    0 sec:  180 sec:      0.000000000 GB:   :       : 1;
        station=Ef:    0 sec:  180 sec:      0.000000000 GB:   : &n    : 1;
        station=Mc:    0 sec:  180 sec:      0.000000000 GB:   : &n    : 1;
        station=O6:    0 sec:  180 sec:      0.000000000 GB:   : &n    : 1;
        station=Tr:    0 sec:  180 sec:      0.000000000 GB:   : &n    : 1;
        station=Hh:    0 sec:  180 sec:      0.000000000 GB:   :       : 1;
        station=Wz:    1 sec:  180 sec:      0.000000000 GB:   : &n    : 1;
    endscan;

In what matters to this section, the relevant components of the scan are its reference epoch (2023y292d15h35m00s), and the time offsets of each of the stations, specified in the second and third columns. A station with offsets ``dt_start`` and ``dt_end``, will record data between ``ref_epoch + dt_start``, and ``ref_epoch + dt_end``. For the example shown above, all the stations will end their recordings 3 minutes after the reference epoch, but ``Wz`` will start one second later than the others.

We say that consecutive scans ``A`` and ``B`` overlap, when a station shared by both of them starts recording in ``B``, before it finishes recording in ``A``

.. math::

    \text{ref_epoch}_A + dt_{\text{end,A}} \gt \text{ref_epoch}_B + dt_{\text{start,B}}

We say that the scans ``A`` and ``B`` are zero-gap, when a station shared by both of them starts recording in ``B`` exactly when it finishes recording in ``A``

.. math::

    \text{ref_epoch}_A + dt_{\text{end,A}} = \text{ref_epoch}_B + dt_{\text{start,B}}


An overlapping between scans results into an error that terminates the execution. However, when zero-gap scans are detected, PRIDE raises a warning, and increments the initial offset of the affected stations by one second.

.. dropdown:: Rationale for scan separation requirement

    The internal architecture of PRIDE is built on top of a small set of concepts, which provide an abstract representation of a VLBI experiment. Three of these concepts are:

    - Scan: Contains a collection of timestamps in which a source was recorded from a station
    - Observation: Collects the timestamps of all the scans associated with a station-source pair
    - Baseline: Collects the timestamps of all the scans associated with a station

    The reason for these groupings is vectorization. Quantities that do not depend on the source, but only on the station and the epoch, (i.e. rotation matrix from ITRF to Topocentric frame) are calculated at the baseline level, resulting into a large array covering the time span of several observations. When an observation object needs the rotation matrix at one of its timestamps, it reads it from the baseline's array instead for calculating it. Our current implementation for the lookup method requires timestamps to be unique at the baseline level.

    One of the precomputed quantities is the derivative of the rotation matrix between ITRF and the local topocentric frame, which is calculated numerically by considering the difference between the matrices at :math:`\pm` 1 seconds with respect to the timestamp. To ensure the uniqueness of timestamps at the baseline level, we need consecutive scans to be separated by more than one second, which is the justification for this requirement.
