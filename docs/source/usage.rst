How to use PRIDE
===================

Once you have installed PRIDE, the easiest way to interact with it is to use the command line interface: ``pride``. For now, this program takes a single, required, argument: the path to a configuration file, which is a YAML file with the following structure.

.. dropdown:: Example configuration file

    .. include:: ./_static/example_config.yaml
        :code: yaml

For normal applications, the only section of this file you will have to interact with is ``Experiment``, which includes the following entries:

vex
    The path to your VEX file. It can be absolute or relative to the configuration file. Examples: ``GR035.vex``, ``ec094a.vix``
target
    A short, lowercase, name for your near-field target. It should be compatible with SPICE. Examples: ``juice``, ``mex``, ``vex``
ignore_stations
    A list of two-letter codes of stations that are listed in the VEX file, but should be ignored during delay estimation. The codes should be capitalized. Example: ``["Cd", "Ww"]``
output_directory
    The path to the directory where the program will save the ``.del`` files it produces as output. The path might be absolute or relative to the configuration file. Example: ``output``

Once the configuration file is ready, you can process the experiment running

.. code-block:: bash

    pride config.yaml

This will create the output directory, and populate it with ``.del`` files, containing geocentric delays, and Doppler phase corrections, for each of the stations.
