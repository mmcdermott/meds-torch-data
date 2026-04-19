"""CLI-level tests for the `MTD_preprocess` entrypoint.

End-to-end pipeline semantics (stage chain + output validation) live in
`test_stages.py::test_pipeline_*`. This file covers the wrapper's CLI contract only:
help output, space-containing paths, and the missing-input error path — plus one
golden-shape smoke test (`test_preprocess_end_to_end`) that validates outputs
produced through the wrapper itself, since the pipeline tests invoke
`MEDS_transform-pipeline` directly and bypass `__main__.py`.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

PREPROCESS_SCRIPT = "MTD_preprocess"

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
    out = subprocess.run(
        [PREPROCESS_SCRIPT, "--help"], shell=False, check=True, capture_output=True, text=True
    )
    assert out.returncode == 0
    assert out.stdout.strip() == HELP_STR.strip()


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


def test_preprocess_end_to_end(simple_static_MEDS: Path):
    """Golden-shape smoke test exercising the full wrapper → pipeline → stages path.

    `test_stages.py::test_pipeline_*` validates golden outputs but invokes
    `MEDS_transform-pipeline` directly, bypassing `MTD_preprocess.__main__`. This test
    closes that gap: it runs the real CLI entrypoint and checks that the produced
    `tokenization/schemas/*.parquet`, `tokenization/event_seqs/*.parquet`, and
    `data/*.nrt` outputs exist, are readable, and carry the expected top-level columns /
    tensor keys. It doesn't pin exact values (the pipeline tests already do that more
    rigorously); the point is to catch wrapper-side regressions — config synthesis,
    subprocess plumbing, config path resolution — that the golden pipeline tests would
    miss.
    """

    with tempfile.TemporaryDirectory() as root_dir:
        cohort_dir = Path(root_dir) / "cohort"

        command = [
            PREPROCESS_SCRIPT,
            f"MEDS_dataset_dir={simple_static_MEDS!s}",
            f"output_dir={cohort_dir!s}",
        ]
        out = subprocess.run(command, shell=False, check=False, capture_output=True, text=True)
        assert out.returncode == 0, (
            f"MTD_preprocess failed (rc={out.returncode}).\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
        )

        # MTD_preprocess lays out per-stage intermediate outputs under `<stage>/` subdirs
        # and the final tensorization outputs under `data/`. (pipeline_tester uses the same
        # convention — see `MEDS_transforms/pytest_plugin.py::pipeline_tester`.)
        schemas = sorted((cohort_dir / "tokenization" / "schemas").rglob("*.parquet"))
        event_seqs = sorted((cohort_dir / "tokenization" / "event_seqs").rglob("*.parquet"))
        nrts = sorted(fp for fp in (cohort_dir / "data").rglob("*.nrt"))
        assert schemas, f"No schema parquets produced under {cohort_dir}/tokenization/schemas/"
        assert event_seqs, f"No event_seqs parquets produced under {cohort_dir}/tokenization/event_seqs/"
        assert nrts, f"No NRT files produced under {cohort_dir}/data/"
        assert len(schemas) == len(event_seqs) == len(nrts), (
            f"Shard-count mismatch: {len(schemas)} schemas, {len(event_seqs)} event_seqs, {len(nrts)} NRTs"
        )

        for fp in schemas:
            df = pl.read_parquet(fp)
            assert {"subject_id", "static_code", "static_numeric_value", "time"}.issubset(df.columns), (
                f"Schema parquet {fp} missing expected columns; got {df.columns}"
            )

        for fp in event_seqs:
            df = pl.read_parquet(fp)
            assert {"subject_id", "code", "time_delta_days", "numeric_value"}.issubset(df.columns), (
                f"Event-seqs parquet {fp} missing expected columns; got {df.columns}"
            )

        for fp in nrts:
            nrt = JointNestedRaggedTensorDict(tensors_fp=fp)
            assert nrt.keys() == {"code", "numeric_value", "time_delta_days"}, (
                f"NRT {fp} has keys {nrt.keys()}, expected {{code, numeric_value, time_delta_days}}"
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
