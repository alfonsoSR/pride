from importlib.resources import files
import yaml
from typing import Any
from pathlib import Path


def internal_file(path: str) -> Path:
    return Path(str(files("pride.data").joinpath(path)))


def external_file(path: str) -> Path:
    return Path.home() / ".pride" / path


def load_catalog(path: str) -> Any:
    return yaml.safe_load(internal_file(path).open())
