"""CLI-level tests for the `MTD_preprocess` entrypoint.

End-to-end pipeline semantics (stage chain + output validation) live in
`test_stages.py::test_pipeline_*`. This file covers the wrapper's CLI contract only:
help output, space-containing paths, and the missing-input error path.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from . import PREPROCESS_SCRIPT

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


def test_preprocess_stage_runner_fp_passthrough(simple_static_MEDS: Path):
    """Covers `MTD_preprocess`'s `stage_runner_fp=` hydra-override plumbing.

    `test_stages.py::test_pipeline_parallel` invokes `MEDS_transform-pipeline` directly via
    `pipeline_tester`, skipping our `__main__.py` wrapper. This test fills the gap by running
    `MTD_preprocess` with a stage runner YAML that references a launcher no installed package
    provides. If the wrapper correctly forwards `--stage_runner_fp`, the inner pipeline fails
    while trying to resolve the launcher (and the failure surfaces the launcher name). If the
    wrapper silently dropped `stage_runner_fp`, the pipeline would instead succeed in the
    default serial mode — a false positive the earlier version of this test would not catch.
    """

    sentinel_launcher = "__mtd_passthrough_sentinel_launcher__"
    with tempfile.TemporaryDirectory() as root_dir:
        runner_fp = Path(root_dir) / "stage_runner.yaml"
        runner_fp.write_text(f"parallelize:\n  launcher: {sentinel_launcher}\n")

        cohort_dir = Path(root_dir) / "cohort"
        command = [
            PREPROCESS_SCRIPT,
            f"MEDS_dataset_dir={simple_static_MEDS!s}",
            f"output_dir={cohort_dir!s}",
            f"stage_runner_fp={runner_fp!s}",
        ]
        out = subprocess.run(command, shell=False, check=False, capture_output=True, text=True)

        combined = out.stdout + out.stderr
        assert out.returncode != 0, (
            "MTD_preprocess should fail when stage_runner_fp references a nonexistent launcher; "
            "a successful run implies the wrapper silently dropped `stage_runner_fp`.\n"
            f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
        )
        assert sentinel_launcher in combined, (
            "Expected failure output to mention the sentinel launcher, confirming the runner "
            "YAML was actually loaded by the inner `MEDS_transform-pipeline` invocation.\n"
            f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
        )


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
