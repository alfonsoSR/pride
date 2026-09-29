from .internal_configuration import internal_catalog_path
from typing import Any
import yaml


def load_catalog(catalog_id: str) -> dict[str, Any]:
    """Loads an internal catalog from its ID

    :param catalog_id: ID of the catalog in config.yaml
    :return: Dictionary with the contents of the catalog
    """

    # Get path to internal catalog
    catalog_path = internal_catalog_path(catalog_id)

    # Load catalog from YAML file
    with catalog_path.open() as buffer:
        catalog = yaml.safe_load(buffer)

    return catalog
