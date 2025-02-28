#!/usr/bin/env python

import logging
import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

from . import ETL_CFG, MAIN_CFG, RESHARD_ETL_CFG, RUNNER_CFG

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(MAIN_CFG.parent), config_name=MAIN_CFG.stem)
def main(cfg: DictConfig):
    """Runs the end-to-end MEDS Extraction pipeline."""

    MEDS_dataset_dir = Path(cfg.MEDS_dataset_dir)
    output_dir = Path(cfg.output_dir)
    stage_runner_fp = cfg.get("stage_runner_fp", None)
    do_reshard = cfg.get("do_reshard", False)

    etl_cfg = RESHARD_ETL_CFG if do_reshard else ETL_CFG

    # Then we construct the rest of the command
    command_parts = [
        f"INPUT_DIR={str(MEDS_dataset_dir.resolve())}",
        f"OUTPUT_DIR={str(output_dir.resolve())}",
        "MEDS_transform-runner",
        f"--config-path={str(RUNNER_CFG.parent.resolve())}",
        f"--config-name={RUNNER_CFG.stem}",
        f"pipeline_config_fp={str(etl_cfg.resolve())}",
    ]
    if int(os.getenv("N_WORKERS", 1)) <= 1:
        logger.info("Running in serial mode as N_WORKERS is not set.")
        command_parts.append("~parallelize")

    if stage_runner_fp:
        command_parts.append(f"stage_runner_fp={stage_runner_fp}")

    if cfg.get("do_overwrite", None) is not None:
        command_parts.append(f"++do_overwrite={cfg.do_overwrite}")

    command_parts.append("'hydra.searchpath=[pkg://MEDS_transforms.configs]'")

    full_cmd = " ".join(command_parts)
    logger.info(f"Running command: {full_cmd}")
    command_out = subprocess.run(full_cmd, shell=True, capture_output=True)

    if command_out.returncode != 0:
        logger.error(f"Command failed with return code {command_out.returncode}.")
        logger.error(f"Command stdout:\n{command_out.stdout.decode()}")
        logger.error(f"Command stderr:\n{command_out.stderr.decode()}")
        raise ValueError(f"Command failed with return code {command_out.returncode}.")
    else:
        logger.debug(f"Command stdout:\n{command_out.stdout.decode()}")


if __name__ == "__main__":
    main()
