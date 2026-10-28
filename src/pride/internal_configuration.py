"""Access to the internal configuration of pride (src/pride/data/config.yaml)

Kept at the top level of the package, with no dependencies on other pride modules
except the logger, so that both `io` and `utils` can import it without a circular
import. `io` re-exports `internal_parameter` and `internal_catalog_path`.
"""

from importlib import resources
from pathlib import Path
import yaml
from .logger import log


def internal_parameter(parameter: str):

    # Load internal parameters from YAML file
    with resources.open_text("pride.data", "config.yaml") as parameters_file:
        configuration = yaml.safe_load(parameters_file)["Configuration"]

    # Check if parameter is present
    if parameter not in configuration:
        log.error(
            f"Parameter '{parameter}' not found in internal configuration."
        )
        exit(1)

    # Return the requested parameter
    return configuration[parameter]


def internal_catalog_path(catalog: str) -> Path:

    # Load internal parameters from YAML file
    with resources.open_text("pride.data", "config.yaml") as parameters_file:
        catalogs = yaml.safe_load(parameters_file)["Catalogues"]

    # Check if file is present
    if catalog not in catalogs:
        log.error(f"Catalog '{catalog}' not found in internal catalogs.")
        exit(1)

    # Return the requested file path
    return Path(str(resources.files("pride.data").joinpath(catalogs[catalog])))
