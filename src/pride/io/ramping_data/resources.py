from ..resources import internal_catalog_path, internal_parameter
from pathlib import Path
from typing import Literal
from ...logger import log

FREQUENCY_RAMPING_FILES_BASE: Path = internal_catalog_path("frequency_ramping")


def get_path_to_ramping_data_file(
    mission: str, ramping_type: Literal["one-way", "three-way"]
) -> Path:

    match ramping_type:
        case "one-way":
            return FREQUENCY_RAMPING_FILES_BASE / f"ramp1w.{mission}"
        case "three-way":
            return FREQUENCY_RAMPING_FILES_BASE / f"ramp3w.{mission}"
        case _:
            log.error(f"Unknown ramping type: {ramping_type}")
            exit(1)
