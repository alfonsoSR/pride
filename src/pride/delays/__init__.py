"""Algorithms for delay estimation"""

from .models import Geometric, AntennaDelays, Ionospheric, Tropospheric
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Delay

DELAY_MODELS: dict[str, type["Delay"]] = {
    Geometric.__name__: Geometric,
    Tropospheric.__name__: Tropospheric,
    Ionospheric.__name__: Ionospheric,
    AntennaDelays.__name__: AntennaDelays,
}

__all__ = [
    "DELAY_MODELS",
    "Geometric",
    "Tropospheric",
    "Ionospheric",
    "AntennaDelays",
]
