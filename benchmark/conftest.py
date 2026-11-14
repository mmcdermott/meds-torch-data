"""Test set-up and fixtures code."""

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from meds import birth_code


def pytest_addoption(parser):
    parser.addoption(
        "--persistent_cache_dir",
        default=None,
        help="Use this local (preserved) directory for persistent caching. Directory must exist.",
    )


@contextmanager
def _test_dir(request):
    if request.config.getoption("--persistent_cache_dir"):
        cache_dir = Path(request.config.getoption("--persistent_cache_dir"))
        if not cache_dir.exists():
            raise FileNotFoundError("Persistent cache directory does not exist.")
        else:
            yield cache_dir
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


def _preprocess(MEDS_dir: Path, cohort_dir: Path) -> None:
    command = [
        "MTD_preprocess",
        f"MEDS_dataset_dir={MEDS_dir!s}",
        f"output_dir={cohort_dir}",
        "do_reshard=True",
    ]

    out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

    error_str = (
        f"Command failed with return code {out.returncode}.\n"
        f"Command stdout:\n{out.stdout.decode()}\n"
        f"Command stderr:\n{out.stderr.decode()}"
    )

    assert out.returncode == 0, error_str


def _add_birth_events(MEDS_dir: Path, seed: int = 0) -> None:
    """Add one `MEDS_BIRTH` event, 20-80 years before each subject's first event, to every shard in place."""
    rng = np.random.default_rng(seed)
    for fp in sorted((MEDS_dir / "data").rglob("*.parquet")):
        df = pl.read_parquet(fp)
        first = (
            df.filter(pl.col("time").is_not_null())
            .group_by("subject_id", maintain_order=True)
            .agg(pl.col("time").min())
        )
        offsets = pl.Series([timedelta(days=365.25 * y) for y in rng.uniform(20, 80, len(first))])
        births = first.with_columns(
            (pl.col("time") - offsets).alias("time"), pl.lit(birth_code).alias("code")
        ).with_columns(
            *[
                pl.lit(None, dtype=df.schema[c]).alias(c)
                for c in df.columns
                if c not in ("subject_id", "time", "code")
            ]
        )
        out = pl.concat([births.select(df.columns).cast(df.schema), df])
        out.sort("subject_id", "time", nulls_last=False, maintain_order=True).write_parquet(fp)


@pytest.fixture(scope="session")
def benchmark_dataset_with_births(request, generated_sample_MEDS: Path) -> Path:
    """The benchmark dataset plus one birth event per subject, so age features can be profiled.

    Kept separate from `benchmark_dataset` so the default benchmark (and its published history) is
    unaffected by the extra events.
    """
    with tempfile.TemporaryDirectory() as tmp, _test_dir(request) as cache_dir:
        MEDS_dir = Path(tmp) / "MEDS"
        shutil.copytree(generated_sample_MEDS, MEDS_dir)
        _add_birth_events(MEDS_dir)
        cohort_dir = Path(cache_dir) / "with_births"
        _preprocess(MEDS_dir, cohort_dir)
        yield cohort_dir


@pytest.fixture(scope="session")
def benchmark_dataset(request, generated_sample_MEDS: Path) -> Path:
    with _test_dir(request) as cohort_dir:
        _preprocess(generated_sample_MEDS, cohort_dir)
        yield Path(cohort_dir)
