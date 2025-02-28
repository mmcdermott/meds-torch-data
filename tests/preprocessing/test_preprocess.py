"""Tests the full, multi-stage pre-processing pipeline. Only checks tokenized and tensorized outputs."""


from pathlib import Path

import polars as pl

from . import assert_df_equal, check_NRT_output
from .test_tensorization import WANT_NRTS
from .test_tokenization import WANT_SCHEMAS


def test_preprocess(tensorized_MEDS_dataset: tuple[Path, Path]):
    cohort_dir = tensorized_MEDS_dataset[1]

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
