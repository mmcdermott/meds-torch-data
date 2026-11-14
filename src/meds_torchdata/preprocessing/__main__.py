import logging
import os
import subprocess
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

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

    # `birth_code_regex` is written into a per-run copy of the pipeline config rather than passed as a
    # `--overrides` entry: the runner both Hydra-parses overrides and shell-joins them into each stage's
    # command, and no single quoting of an arbitrary regex survives both. The copy lives in a temp dir
    # (not `output_dir`, which may contain spaces that the runner's shell-joined commands can't handle).
    with tempfile.TemporaryDirectory() as tmp_cfg_dir:
        pipeline_cfg_fp = write_pipeline_config(etl_cfg, Path(tmp_cfg_dir), cfg.get("birth_code_regex", None))
        run_pipeline(pipeline_cfg_fp, cfg, MEDS_dataset_dir, output_dir, stage_runner_fp)


def write_pipeline_config(etl_cfg: Path, out_dir: Path, birth_code_regex: str | None) -> Path:
    """Write a copy of the pipeline config with `birth_code_regex` set, returning its path.

    Interpolations (e.g., `${oc.env:INPUT_DIR}`) are preserved unresolved.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = write_pipeline_config(ETL_CFG, Path(tmpdir), "^DOB$")
        ...     cfg = OmegaConf.load(fp)
        ...     print(cfg.birth_code_regex, cfg.input_dir)
        ...     print(OmegaConf.to_container(cfg.stages[4]))
        ^DOB$ ${oc.env:INPUT_DIR}
        {'tokenization': {'birth_code_regex': '${birth_code_regex}'}}
    """
    pipeline_cfg = OmegaConf.load(etl_cfg)
    if birth_code_regex is not None:
        pipeline_cfg.birth_code_regex = birth_code_regex
    out_fp = out_dir / Path(str(etl_cfg)).name
    OmegaConf.save(pipeline_cfg, out_fp, resolve=False)
    return out_fp


def run_pipeline(
    pipeline_cfg_fp: Path,
    cfg: DictConfig,
    MEDS_dataset_dir: Path,
    output_dir: Path,
    stage_runner_fp: str | None,
) -> None:
    """Run `MEDS_transform-pipeline` on `pipeline_cfg_fp`, raising if it fails."""
    cmd = ["MEDS_transform-pipeline", str(pipeline_cfg_fp.resolve())]

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
