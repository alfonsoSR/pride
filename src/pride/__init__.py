"""
PRIDE Reference
=================

Welcome to the API reference of PRIDE!
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pride")
except PackageNotFoundError:
    pass
