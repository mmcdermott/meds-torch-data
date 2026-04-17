"""Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `subject_id`, `time`, `code`, `numeric_value`.
"""

import logging
from pathlib import Path

import polars as pl
from MEDS_transforms.dataframe import write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.mapreduce.shard_iteration import shard_iterator
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf

from .._stage_example import MTDStageExample

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24.0

logger = logging.getLogger(__name__)


def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """This function fills infinite and null values with NaN.

    This enables the downstream functions to naturally tensorize data into numpy or Torch tensors.

    Args:
        col: The input column.

    Returns:
        A `pl.Expr` object that fills infinite and null values with NaN.

    Examples:
        >>> print(fill_to_nans("value"))
        .when([(col("value").is_infinite()) |
               (col("value").is_null())]).then(dyn float: NaN).otherwise(col("value"))
        >>> print(fill_to_nans(pl.col("time_delta")))
        .when([(col("time_delta").is_infinite()) |
               (col("time_delta").is_null())]).then(dyn float: NaN).otherwise(col("time_delta"))
        >>> df = pl.DataFrame({"value": [1.0, float("inf"), None, -float("inf"), 2.0]})
        >>> df.select(fill_to_nans("value").alias("value"))["value"].to_list()
        [1.0, nan, nan, nan, 2.0]
    """

    if isinstance(col, str):
        col = pl.col(col)

    return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)


def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """This function splits the input data into static and dynamic data.

    Static data is data that has a null time, and dynamic data is everything else.

    Args:
        df: The input data.

    Returns:
        A tuple of two `pl.LazyFrame` objects, the first being the static data and the second being the
        dynamic data.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0]
        ... }).lazy()
        >>> static, dynamic = split_static_and_dynamic(df)
        >>> static.collect()
        shape: (2, 3)
        ┌────────────┬──────┬───────────────┐
        │ subject_id ┆ code ┆ numeric_value │
        │ ---        ┆ ---  ┆ ---           │
        │ i64        ┆ i64  ┆ f64           │
        ╞════════════╪══════╪═══════════════╡
        │ 1          ┆ 100  ┆ 1.0           │
        │ 2          ┆ 200  ┆ 3.0           │
        └────────────┴──────┴───────────────┘
        >>> dynamic.collect()
        shape: (2, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ f64           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 101  ┆ 2.0           │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 201  ┆ 4.0           │
        └────────────┴─────────────────────┴──────┴───────────────┘
    """

    static = df.filter(pl.col("time").is_null()).drop("time")
    dynamic = df.filter(pl.col("time").is_not_null())
    return static, dynamic


def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """This function extracts static data and schema information (sequence of subject unique times).

    Args:
        df: The input data.

    Returns:
        A `pl.LazyFrame` object containing the static data and the unique times of the subject, grouped
        by subject as lists, in the same order as the subject IDs occurred in the original file. Each
        subject also carries a `measurements_per_event` list, parallel to the `time` list, giving the
        number of raw measurements (rows in the source dataframe) at each unique timestamp. This column
        is needed by `STEP_THROUGH` sampling in `BatchMode.SM` to map measurement-level window ends back
        to event-level indices.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2, 2],
        ...     "time": [
        ...         None, datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...         None, datetime(2021, 1, 2), datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 103, 200, 201, 202],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ... }).lazy()
        >>> df = extract_statics_and_schema(df).collect()
        >>> df.drop("time", "measurements_per_event")
        shape: (2, 4)
        ┌────────────┬─────────────┬──────────────────────┬─────────────────────┐
        │ subject_id ┆ static_code ┆ static_numeric_value ┆ start_time          │
        │ ---        ┆ ---         ┆ ---                  ┆ ---                 │
        │ i64        ┆ list[i64]   ┆ list[f64]            ┆ datetime[μs]        │
        ╞════════════╪═════════════╪══════════════════════╪═════════════════════╡
        │ 1          ┆ [100]       ┆ [1.0]                ┆ 2021-01-01 00:00:00 │
        │ 2          ┆ [200]       ┆ [5.0]                ┆ 2021-01-02 00:00:00 │
        └────────────┴─────────────┴──────────────────────┴─────────────────────┘
        >>> df.select("subject_id", "time", "measurements_per_event").explode(
        ...     "time", "measurements_per_event"
        ... )
        shape: (3, 3)
        ┌────────────┬─────────────────────┬────────────────────────┐
        │ subject_id ┆ time                ┆ measurements_per_event │
        │ ---        ┆ ---                 ┆ ---                    │
        │ i64        ┆ datetime[μs]        ┆ u32                    │
        ╞════════════╪═════════════════════╪════════════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 2                      │
        │ 1          ┆ 2021-01-13 00:00:00 ┆ 1                      │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 2                      │
        └────────────┴─────────────────────┴────────────────────────┘
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2, 2],
        ...     "time": [
        ...         datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...         datetime(2020, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 103, 200, 201, 202],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ... }).lazy()
        >>> df = extract_statics_and_schema(df).collect()
        >>> df.drop("time", "measurements_per_event")
        shape: (2, 4)
        ┌────────────┬─────────────┬──────────────────────┬─────────────────────┐
        │ subject_id ┆ static_code ┆ static_numeric_value ┆ start_time          │
        │ ---        ┆ ---         ┆ ---                  ┆ ---                 │
        │ i64        ┆ list[i64]   ┆ list[f64]            ┆ datetime[μs]        │
        ╞════════════╪═════════════╪══════════════════════╪═════════════════════╡
        │ 1          ┆ null        ┆ null                 ┆ 2020-01-01 00:00:00 │
        │ 2          ┆ null        ┆ null                 ┆ 2020-01-01 00:00:00 │
        └────────────┴─────────────┴──────────────────────┴─────────────────────┘
        >>> df.select("subject_id", "time", "measurements_per_event").explode(
        ...     "time", "measurements_per_event"
        ... )
        shape: (5, 3)
        ┌────────────┬─────────────────────┬────────────────────────┐
        │ subject_id ┆ time                ┆ measurements_per_event │
        │ ---        ┆ ---                 ┆ ---                    │
        │ i64        ┆ datetime[μs]        ┆ u32                    │
        ╞════════════╪═════════════════════╪════════════════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ 1                      │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 2                      │
        │ 1          ┆ 2021-01-13 00:00:00 ┆ 1                      │
        │ 2          ┆ 2020-01-01 00:00:00 ┆ 1                      │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 2                      │
        └────────────┴─────────────────────┴────────────────────────┘

        Source order is preserved across the full-outer join, not re-sorted by `subject_id`.
        The input below has subject IDs `[7, 3, 5, 2, 7, 5]` — non-monotonic in either direction,
        with duplicate subjects that differ in whether they carry a static row — and the output
        preserves the first-occurrence order `[7, 3, 5, 2]`. Static-only subjects (here `7` and
        `5`) carry non-null `static_code`; dynamic-only subjects (`3` and `2`) carry `null`.
        (Regression test: polars 1.39+ stopped preserving pre-join ordering for `how="full"`.)

        >>> df = pl.DataFrame({
        ...     "subject_id": [7, 3, 5, 2, 7, 5],
        ...     "time": [
        ...         None, datetime(2021, 1, 1), None, datetime(2021, 2, 1),
        ...         datetime(2021, 3, 1), datetime(2021, 4, 1),
        ...     ],
        ...     "code": [700, 300, 500, 200, 701, 501],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ... }).lazy()
        >>> extract_statics_and_schema(df).collect().select("subject_id", "static_code")
        shape: (4, 2)
        ┌────────────┬─────────────┐
        │ subject_id ┆ static_code │
        │ ---        ┆ ---         │
        │ i64        ┆ list[i64]   │
        ╞════════════╪═════════════╡
        │ 7          ┆ [700]       │
        │ 3          ┆ null        │
        │ 5          ┆ [500]       │
        │ 2          ┆ null        │
        └────────────┴─────────────┘
    """

    static, dynamic = split_static_and_dynamic(df)

    # This collects static data by subject ID and stores only (as a list) the codes and numeric values.
    static_by_subject = static.group_by("subject_id", maintain_order=True).agg(
        pl.col("code").alias("static_code"),
        pl.col("numeric_value").alias("static_numeric_value"),
    )

    # Collect unique times per subject, along with the number of measurements at each timestamp (needed
    # by STEP_THROUGH sampling in SM mode to map measurement-level window ends back to event indices
    # without re-loading the dynamic tensor). The two-step group_by preserves source order, so the
    # `time` and `measurements_per_event` lists line up element-for-element.
    schema_by_subject = (
        dynamic.group_by(["subject_id", "time"], maintain_order=True)
        .agg(pl.len().alias("measurements_per_event"))
        .group_by("subject_id", maintain_order=True)
        .agg(
            pl.col("time").min().alias("start_time"),
            pl.col("time"),
            pl.col("measurements_per_event"),
        )
    )

    # polars 1.39+ no longer preserves pre-join row ordering for `how="full"`, so the source-order
    # guarantee in the docstring can't rely on the `maintain_order=True` group_bys alone. Attach an
    # explicit first-occurrence-in-source index and sort by it after the join.
    subject_order = df.select("subject_id").unique(maintain_order=True).with_row_index("_subject_order")

    return (
        static_by_subject.join(schema_by_subject, on="subject_id", how="full", coalesce=True)
        .join(subject_order, on="subject_id", how="left")
        .sort("_subject_order")
        .drop("_subject_order")
    )


def extract_seq_of_subject_events(df: pl.LazyFrame) -> pl.LazyFrame:
    """This function extracts sequences of subject events, which are sequences of measurements.

    The result of this can be naturally tensorized into a `JointNestedRaggedTensorDict` object.

    Args:
        df: The input data.

    Returns:
        A `pl.LazyFrame` object containing the sequences of subject events, with the following columns:
            - `subject_id`: The subject ID.
            - `time_delta_days`: The time delta in days, as a list of floats (ragged).
            - `code`: The code, as a list of lists of ints (ragged in both levels).
            - `numeric_value`: The numeric value as a list of lists of floats (ragged in both levels).

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2, 2],
        ...     "time": [
        ...         None, datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...         None, datetime(2021, 1, 2), datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 103, 200, 201, 202],
        ...     "numeric_value": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=pl.Float32)
        ... }).lazy()
        >>> extract_seq_of_subject_events(df).collect()
        shape: (2, 4)
        ┌────────────┬─────────────────┬─────────────────────┬─────────────────────┐
        │ subject_id ┆ time_delta_days ┆ code                ┆ numeric_value       │
        │ ---        ┆ ---             ┆ ---                 ┆ ---                 │
        │ i64        ┆ list[f32]       ┆ list[list[i64]]     ┆ list[list[f32]]     │
        ╞════════════╪═════════════════╪═════════════════════╪═════════════════════╡
        │ 1          ┆ [NaN, 12.0]     ┆ [[101, 102], [103]] ┆ [[2.0, 3.0], [4.0]] │
        │ 2          ┆ [NaN]           ┆ [[201, 202]]        ┆ [[6.0, 7.0]]        │
        └────────────┴─────────────────┴─────────────────────┴─────────────────────┘
    """

    _, dynamic = split_static_and_dynamic(df)

    time_delta_days_expr = (pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float32)

    return (
        dynamic.group_by("subject_id", "time", maintain_order=True)
        .agg(pl.col("code").name.keep(), fill_to_nans("numeric_value").name.keep())
        .group_by("subject_id", maintain_order=True)
        .agg(
            fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
            "code",
            "numeric_value",
        )
    )


@Stage.register(is_metadata=False, example_class=MTDStageExample)
def main(cfg: DictConfig):
    """Tokenizes the dataset in accordance with the aggregated code metadata.

    Note that you _must_ run several other stages first, pending tokenization mode, including
    `fit_vocabulary_indices` and `aggregate_code_metadata` with the appropriate aggregations. See the
    `stage_configs` for arg options.
    """

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    if train_only := cfg.stage_cfg.get("train_only", False):
        raise ValueError(f"train_only={train_only} is not supported for this stage.")
    shards_single_output, _ = shard_iterator(cfg)

    for in_fp, out_fp in shards_single_output:
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path

        logger.info(f"Tokenizing {in_fp.resolve()!s} into schemas at {schema_out_fp.resolve()!s}")

        rwlock_wrap(
            in_fp,
            schema_out_fp,
            pl.scan_parquet,
            write_df,
            extract_statics_and_schema,
            do_overwrite=cfg.do_overwrite,
        )

        logger.info(f"Tokenizing {in_fp.resolve()!s} into event_seqs at {event_seq_out_fp.resolve()!s}")

        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_df,
            extract_seq_of_subject_events,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")
