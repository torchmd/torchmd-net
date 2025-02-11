from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torchmd-net")
except PackageNotFoundError:
    # package is not installed
    pass
