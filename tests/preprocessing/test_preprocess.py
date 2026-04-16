"""Tests the full, multi-stage pre-processing pipeline.

Only checks tokenized and tensorized outputs.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from . import PREPROCESS_SCRIPT, assert_df_equal, check_NRT_output
from .test_tensorization import WANT_NRTS
from .test_tokenization import WANT_SCHEMAS

HELP_STR = """
== MTD_preprocess ==

MTD_preprocess is a command line tool for pre-processing MEDS data for use with meds_torchdata.

== Config ==

This is the config generated for this run:

MEDS_dataset_dir: ???
output_dir: ???
stage_runner_fp: null
do_overwrite: false
do_reshard: false
log_dir: ${output_dir}/.logs

You can override everything using the hydra `key=value` syntax; for example:

MTD_preprocess MEDS_dataset_dir=/path/to/dataset output_dir=/path/to/output do_overwrite=True
"""


def test_preprocess_help():
    out = subprocess.run(f"{PREPROCESS_SCRIPT} --help", shell=True, check=True, capture_output=True)
    assert out.returncode == 0
    assert out.stdout.decode().strip() == HELP_STR.strip()


def test_preprocess(tensorized_MEDS_dataset: Path):
    cohort_dir = tensorized_MEDS_dataset

    cohort_dir_contents = list(cohort_dir.rglob("*.parquet")) + list(cohort_dir.rglob("*.nrt"))
    cohort_dir_contents_str = "\n".join(f"  - {f.relative_to(cohort_dir)}" for f in cohort_dir_contents)

    for shard, want_schema in WANT_SCHEMAS.items():
        fp = cohort_dir / f"tokenization/{shard}.parquet"
        err_str = f"Expected output file {fp} not found. Directory contents:\n" + cohort_dir_contents_str

        assert fp.exists(), err_str
        got_schema = pl.read_parquet(fp)
        assert_df_equal(got_schema, want_schema, check_column_order=False)

    for shard, want_NRT in WANT_NRTS.items():
        fp = cohort_dir / f"data/{shard}"
        err_str = f"Expected output file {fp} not found. Directory contents:\n" + cohort_dir_contents_str

        assert fp.exists(), err_str
        check_NRT_output(fp, want_NRT, f"{shard} NRT differs!")


def test_preprocess_path_with_spaces(simple_static_MEDS: Path):
    """MEDS_dataset_dir / output_dir containing spaces must not break the inner ETL subprocess.

    The outer `MTD_preprocess` hydra invocation is run without `shell=True` so the spaces in the
    override values reach hydra intact; the regression this guards against is the *inner*
    `MEDS_transform-pipeline` subprocess corrupting space-containing paths when they pass
    through via `INPUT_DIR` / `OUTPUT_DIR`.
    """

    with tempfile.TemporaryDirectory() as root_dir:
        spaced_src = Path(root_dir) / "input with space"
        shutil.copytree(simple_static_MEDS, spaced_src)

        spaced_out = Path(root_dir) / "output with space"

        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={spaced_src.resolve()!s}",
            f"output_dir={spaced_out.resolve()!s}",
        ]

        out = subprocess.run(command, shell=False, check=False, capture_output=True, text=True)

        assert out.returncode == 0, (
            f"Preprocess failed on a path with spaces (rc={out.returncode}).\n"
            f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
        )

        assert any(spaced_out.rglob("*.parquet")), "No parquet outputs produced."
        assert any(spaced_out.rglob("*.nrt")), "No NRT outputs produced."


def test_preprocess_error_case():
    with tempfile.TemporaryDirectory() as root_dir:
        non_existent_dir = Path(root_dir) / "non_existent_dir"
        cohort_dir = Path(root_dir) / "cohort_dir"

        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={non_existent_dir.resolve()!s}",
            f"output_dir={cohort_dir.resolve()!s}",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        assert out.returncode == 1
