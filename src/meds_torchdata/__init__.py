from importlib.metadata import PackageNotFoundError, version

__package_name__ = "meds_torchdata"

try:
    __version__ = version(__package_name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
