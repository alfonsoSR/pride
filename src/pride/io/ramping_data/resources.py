from ..resources import load_catalog, internal_file
from pathlib import Path
from typing import Literal
from ...logger import log

FREQUENCY_RAMPING_FILES_BASE: str = load_catalog("config.yaml")["Catalogues"][
    "frequency_ramping"
]


def get_path_to_ramping_data_file(
    mission: str, ramping_type: Literal["one-way", "three-way"]
) -> Path:

    match ramping_type:
        case "one-way":
            return internal_file(f"{FREQUENCY_RAMPING_FILES_BASE}1w.{mission}")
        case "three-way":
            return internal_file(f"{FREQUENCY_RAMPING_FILES_BASE}3w.{mission}")
        case _:
            log.error(f"Unknown ramping type: {ramping_type}")
            exit(1)
