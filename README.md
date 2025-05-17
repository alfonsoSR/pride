# PRIDE: Planetary Radio-Interferometry Delay Estimator
[![Tests](https://github.com/alfonsoSR/pride/actions/workflows/tests.yaml/badge.svg)](https://github.com/alfonsoSR/pride/actions/workflows/tests.yaml)
[![Docs](https://img.shields.io/readthedocs/pypride/latest.svg?logo=read%20the%20docs&logoColor=white&label=Docs&version=stable)](https://pypride.readthedocs.io/en/stable/?badge=stable)

## Installation

```
git clone https://github.com/alfonsoSR/pride.git
cd pride
pip3 install .
```

## Usage

PRIDE will be updated with a command line interface in the near future. For the time being, the easiest way to use it is to take the code in `examples/main.py` as reference, and perform the following modifications in the `Experiment` section of `examples/config.yaml`:
- Set `vex` to the path to your VEX file
- Set `target` to a name of your target that can be recognized by SPICE
- Set `output_directory` to the path of your desired output directory. It will be created if it does not exist.

That's it! If you now run `examples/main.py` you will get the `.del` files in your chosen output directory.

## Example

You can find an example application under `examples`. You should be able to run the program `main.py` to calculate the delays for the GR035 experiment. Binary output files will be saved to `examples/output-gr035`. You can use the script `examine_results.py` to plot the delays in these files.

The first time you run the program it will take a little bit because it has to download kernels and data files. If you already have the data files and kernels available, the execution should take about 3 minutes.

If you want to turn off the DEBUG logs, you can set the log level to `INFO` in `main.py`

## Credit

- Original author: [Dimitry Duev](https://github.com/dmitryduev)
- Original Python 3 Version: [Guifre Molera](https://gitlab.com/gofrito)
- 2025 Version & Current Maintainer: [Alfonso Sánchez Rodríguez](https://github.com/alfonsoSR)
