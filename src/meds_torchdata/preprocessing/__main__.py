import logging
import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

from . import ETL_CFG, MAIN_CFG, RESHARD_ETL_CFG

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(MAIN_CFG.parent), config_name=MAIN_CFG.stem)
def main(cfg: DictConfig):
    """Runs the end-to-end MEDS Extraction pipeline.

    MEDS-Transforms 0.6.x replaced the previous Hydra-driven `~parallelize` override with a
    config-only parallelization model: the `parallelize` block must live inside a stage runner
    YAML or inside `additional_params.parallelize` in the pipeline config, and the `joblib`
    launcher requires the optional `hydra-joblib-launcher` package. To keep this entrypoint
    honest about what it can offer, we no longer consult the `N_WORKERS` env var — users who
    want parallel execution pass their own `stage_runner_fp` (see the test suite's
    `PARALLEL_STAGE_RUNNER_YAML` for a minimal joblib example).
    """

    MEDS_dataset_dir = Path(cfg.MEDS_dataset_dir)
    output_dir = Path(cfg.output_dir)
    stage_runner_fp = cfg.get("stage_runner_fp", None)
    do_reshard = cfg.get("do_reshard", False)

    etl_cfg = RESHARD_ETL_CFG if do_reshard else ETL_CFG

    cmd = ["MEDS_transform-pipeline", str(etl_cfg.resolve())]

    if stage_runner_fp:
        cmd.extend(["--stage_runner_fp", str(stage_runner_fp)])

    overrides: list[str] = []
    if cfg.get("do_overwrite", None) is not None:
        overrides.append(f"do_overwrite={cfg.do_overwrite}")

    if overrides:
        cmd.append("--overrides")
        cmd.extend(overrides)

    # `MEDS_transform-pipeline` reads `INPUT_DIR` / `OUTPUT_DIR` from the environment rather than
    # from CLI flags, so we pass them via `env=` to avoid the `shell=True` / string-joining path
    # that would otherwise corrupt dataset paths containing spaces or shell metacharacters.
    env = {
        **os.environ,
        "INPUT_DIR": str(MEDS_dataset_dir.resolve()),
        "OUTPUT_DIR": str(output_dir.resolve()),
    }
    logger.info(f"Running command: INPUT_DIR={env['INPUT_DIR']} OUTPUT_DIR={env['OUTPUT_DIR']} {cmd}")
    command_out = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if command_out.returncode != 0:
        logger.error(f"Command failed with return code {command_out.returncode}.")
        logger.error(f"Command stdout:\n{command_out.stdout}")
        logger.error(f"Command stderr:\n{command_out.stderr}")
        raise ValueError(f"Command failed with return code {command_out.returncode}.")
    else:
        logger.debug(f"Command stdout:\n{command_out.stdout}")
