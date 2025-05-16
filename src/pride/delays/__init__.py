"""Algorithms for delay estimation"""

from .models import Geometric, AntennaDelays, Ionospheric, Tropospheric
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Delay

DELAY_MODELS: dict[str, type["Delay"]] = {
    Geometric.name: Geometric,
    Tropospheric.name: Tropospheric,
    Ionospheric.name: Ionospheric,
    AntennaDelays.name: AntennaDelays,
}

__all__ = [
    "DELAY_MODELS",
    "Geometric",
    "Tropospheric",
    "Ionospheric",
    "AntennaDelays",
]
