"""Algorithms to model geophysical displacements of stations"""

from .models import SolidTide, OceanLoading, PoleTide
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Displacement

DISPLACEMENT_MODELS: dict[str, type["Displacement"]] = {
    SolidTide.__name__: SolidTide,
    OceanLoading.__name__: OceanLoading,
    PoleTide.__name__: PoleTide,
}

__all__ = ["DISPLACEMENT_MODELS", "SolidTide", "OceanLoading", "PoleTide"]
