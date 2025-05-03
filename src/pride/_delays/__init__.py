raise DeprecationWarning("Deprecated module kept for reference during development")

from .models import Geometric, Tropospheric, Ionospheric, ThermalDeformation

DELAY_MODELS = {
    Geometric.name: Geometric,
    Tropospheric.name: Tropospheric,
    Ionospheric.name: Ionospheric,
    ThermalDeformation.name: ThermalDeformation,
}

__all__ = [
    "DELAY_MODELS",
    "Geometric",
    "Tropospheric",
    "Ionospheric",
    "ThermalDeformation",
]
