from .models import SolidTide, OceanLoading, PoleTide
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Displacement

DISPLACEMENT_MODELS: dict[str, type["Displacement"]] = {
    SolidTide.name: SolidTide,
    OceanLoading.name: OceanLoading,
    PoleTide.name: PoleTide,
}

__all__ = ["DISPLACEMENT_MODELS"]
