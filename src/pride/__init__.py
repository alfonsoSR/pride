from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pride")
except PackageNotFoundError:
    pass
