from importlib.resources import files

from .. import __package_name__

MAIN_CFG = files(__package_name__).joinpath("preprocessing/configs/main.yaml")
ETL_CFG = files(__package_name__).joinpath("preprocessing/configs/_tensorize.yaml")
RESHARD_ETL_CFG = files(__package_name__).joinpath("preprocessing/configs/_reshard_then_tensorize.yaml")
RUNNER_CFG = files(__package_name__).joinpath("preprocessing/configs/runner.yaml")

__all__ = ["ETL_CFG", "MAIN_CFG"]
