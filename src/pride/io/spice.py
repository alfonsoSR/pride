from .resources import load_catalog
from typing import Any
from ..logger import log
from pathlib import Path

ESA_SPICE = "https://spiftp.esac.esa.int/data/SPICE"
NASA_SPICE = "https://naif.jpl.nasa.gov/pub/naif"


def get_target_information(name: str) -> dict[str, Any]:
    """Get target information from spacecraft catalog

    :param name: Name of the target
    :return: Dictionary containing target information
    """

    name = name.upper()
    catalog = load_catalog("spacecraft.yaml")
    out: dict[str, Any] | None = None

    for target in catalog.values():
        if name in target["names"]:
            out = target
            break
    if out is None:
        log.error(f"Target {name} not found in spacecraft catalog")
        exit(1)

    return out
