"""Algorithms for Doppler estimation"""

from .models import Dop
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Doppler

DOPPLER_MODELS: dict[str, type["Doppler"]] = {
    Dop.__name__: Dop,
}

__all__ = [
    "DOPPLER_MODELS",
    "Dop",
]
