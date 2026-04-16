import logging
import os
import subprocess
import tempfile
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig

from . import ETL_CFG, MAIN_CFG, RESHARD_ETL_CFG

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(MAIN_CFG.parent), config_name=MAIN_CFG.stem)
def main(cfg: DictConfig):
    """Runs the end-to-end MEDS Extraction pipeline."""

    MEDS_dataset_dir = Path(cfg.MEDS_dataset_dir)
    output_dir = Path(cfg.output_dir)
    stage_runner_fp = cfg.get("stage_runner_fp", None)
    do_reshard = cfg.get("do_reshard", False)

    etl_cfg = RESHARD_ETL_CFG if do_reshard else ETL_CFG

    # MEDS-Transforms 0.6.x switched `MEDS_transform-pipeline` from a Hydra app
    # to an argparse CLI. The new shape is:
    #   MEDS_transform-pipeline <pipeline_config_fp>
    #     [--stage_runner_fp STAGE_RUNNER_FP]
    #     [--overrides KEY=VALUE ...]
    # Parallelization is now configured via the `parallelize` block inside the
    # stage runner yaml (or the pipeline config's `additional_params`) rather
    # than via a Hydra override. We synthesize the stage runner yaml at call
    # time so we can gate the `parallelize` block on N_WORKERS exactly as the
    # old `~parallelize` override used to.
    n_workers = int(os.getenv("N_WORKERS", 1))

    cmd = ["MEDS_transform-pipeline", str(etl_cfg.resolve())]

    user_stage_runner_fp = stage_runner_fp
    synthesized_runner_path: Path | None = None
    if user_stage_runner_fp:
        cmd.extend(["--stage_runner_fp", str(user_stage_runner_fp)])
    elif n_workers > 1:
        synthesized = {"parallelize": {"n_workers": n_workers, "launcher": "joblib"}}
        tmp_dir = Path(tempfile.mkdtemp(prefix="mtd_stage_runner_"))
        synthesized_runner_path = tmp_dir / "stage_runner.yaml"
        synthesized_runner_path.write_text(yaml.safe_dump(synthesized))
        cmd.extend(["--stage_runner_fp", str(synthesized_runner_path)])
    else:
        logger.info(f"Running in serial mode (n_workers={n_workers} <= 1).")

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
    try:
        command_out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    finally:
        if synthesized_runner_path is not None:
            try:
                synthesized_runner_path.unlink(missing_ok=True)
                synthesized_runner_path.parent.rmdir()
            except OSError:
                pass

    if command_out.returncode != 0:
        logger.error(f"Command failed with return code {command_out.returncode}.")
        logger.error(f"Command stdout:\n{command_out.stdout}")
        logger.error(f"Command stderr:\n{command_out.stderr}")
        raise ValueError(f"Command failed with return code {command_out.returncode}.")
    else:
        logger.debug(f"Command stdout:\n{command_out.stdout}")
