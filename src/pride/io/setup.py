import yaml
from pathlib import Path
from typing import Any
from .resources import internal_file, load_catalog


class Setup:
    """Delay calculation setup

    :param config_file: Path to configuration file
    """

    def __init__(self, config_file: str) -> None:

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Paths to internal and external files
        external = Path().home() / ".pride"

        # Internal configuration
        internal_config = load_catalog("config.yaml")
        self.catalogues: dict[str, Any] = internal_config["Catalogues"]
        self.internal: dict[str, Any] = internal_config["Configuration"]

        # User configuration
        self.general: dict[str, Any] = config["Experiment"]

        self.resources: dict[str, Path] = config["Resources"]
        for key, val in self.resources.items():
            self.resources[key] = external / val

        self.displacements: dict[str, bool] = config["Displacements"]

        self.delays: dict[str, dict[str, Any]] = config["Delays"]
        for delay in self.delays.values():
            delay["data"] = external / delay["data"]

        return None
