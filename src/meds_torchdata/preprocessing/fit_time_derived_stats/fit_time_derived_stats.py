"""Fits training-set statistics for time-derived measurements (currently, subject age).

Time-derived measurements are values computed from a subject's timeline rather than read from the
data, such as the subject's age at each event. `MEDSPytorchDataset` computes them on the fly (see
`MEDSTorchDataConfig.age_token_mode` and `MEDSTorchDataConfig.include_age`), but normalizing or
quantizing them consistently requires statistics of their training distribution. This metadata stage
computes those statistics once, at preprocessing time, from the tokenization stage's subject schemas
(which carry each subject's event `time`s and `birth_time`).

The output, `time_derived_measurement_stats.parquet` in the stage's metadata output directory, has one
row per time-derived measurement. As this is a metadata stage, every file in the metadata input
directory (e.g., `codes.parquet`) is also passed through to the metadata output directory unchanged.
"""

import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from meds import train_split
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from .._stage_example import MTDStageExample

logger = logging.getLogger(__name__)

STATS_FN = "time_derived_measurement_stats.parquet"

DAYS_PER_YEAR = 365.25
MICROSECONDS_PER_YEAR = DAYS_PER_YEAR * 24 * 60 * 60 * 1e6

# Ages are accumulated into a fixed-width histogram (1 day per bin) so that quantiles can be computed
# across arbitrarily many shards in constant memory. Ages outside this range are clipped into the
# boundary bins, which only affects the extreme quantiles (the exact min / max are tracked separately).
AGE_HIST_MIN_YEARS = -2.0
AGE_HIST_MAX_YEARS = 150.0
AGE_HIST_BIN_WIDTH_YEARS = 1.0 / DAYS_PER_YEAR

STATS_SCHEMA = {
    "measurement": pl.String,
    "unit": pl.String,
    "n_observations": pl.Int64,
    "mean": pl.Float64,
    "std": pl.Float64,
    "quantiles": pl.List(pl.Float64),
}


def non_birth_event_ages(schema_df: pl.LazyFrame) -> pl.LazyFrame:
    """Returns the age (in years) at every non-birth event of every subject with a known birth time.

    The birth event itself is excluded: birth typically precedes every other event by years or decades,
    so including it would add a spurious age-0 observation per subject and distort the statistics.

    Args:
        schema_df: A tokenization-stage subject schema, with (at least) a `time` list column of the
            subject's event times and a `birth_time` column.

    Returns:
        A single-column (`age`) `LazyFrame`.

    Examples:
        >>> from datetime import datetime
        >>> schema_df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "time": [
        ...         [datetime(2000, 1, 1), datetime(2010, 1, 1), datetime(2020, 1, 1)],
        ...         [datetime(2005, 1, 1)],
        ...         [datetime(1999, 1, 1), datetime(2001, 1, 1)],
        ...     ],
        ...     "birth_time": [datetime(2000, 1, 1), None, datetime(2000, 1, 1)],
        ... }).lazy()
        >>> non_birth_event_ages(schema_df).collect()["age"].round(3).to_list()
        [10.001, 20.0, -0.999, 1.002]
    """
    return (
        # Casting both to a common unit also guards against an all-null `birth_time` column being
        # stored with polars' `Null` dtype, which cannot be subtracted from a datetime.
        schema_df.select(
            pl.col("time").cast(pl.List(pl.Datetime("us"))), pl.col("birth_time").cast(pl.Datetime("us"))
        )
        .filter(pl.col("birth_time").is_not_null())
        .explode("time")
        .filter(pl.col("time") != pl.col("birth_time"))
        .select(
            ((pl.col("time") - pl.col("birth_time")).dt.total_microseconds() / MICROSECONDS_PER_YEAR).alias(
                "age"
            )
        )
    )


class _RunningStats:
    """Accumulates count, sum, sum of squares, extrema, and a fixed-width histogram over batches."""

    def __init__(self, hist_min: float, hist_max: float, bin_width: float):
        self.hist_min = hist_min
        self.bin_width = bin_width
        self.n_bins = int(np.ceil((hist_max - hist_min) / bin_width))
        self.counts = np.zeros(self.n_bins, dtype=np.int64)
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return
        self.n += len(values)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        self.min = min(self.min, float(values.min()))
        self.max = max(self.max, float(values.max()))
        bins = np.clip(((values - self.hist_min) / self.bin_width).astype(np.int64), 0, self.n_bins - 1)
        self.counts += np.bincount(bins, minlength=self.n_bins)

    def quantiles(self, n_quantiles: int) -> list[float]:
        """Quantiles at `n_quantiles` evenly spaced levels in [0, 1], interpolated within histogram bins.

        The 0 and 1 quantiles are the exact observed min and max; interior quantiles are accurate to
        within one histogram bin.

        Examples:
            >>> stats = _RunningStats(0.0, 10.0, 0.5)
            >>> stats.update(np.array([1.0, 2.0, 3.0, 4.0]))
            >>> stats.quantiles(5)
            [1.0, 1.5, 2.5, 3.5, 4.0]
        """
        cum = np.concatenate([[0], np.cumsum(self.counts)])
        edges = self.hist_min + self.bin_width * np.arange(self.n_bins + 1)
        out = []
        for level in np.linspace(0.0, 1.0, n_quantiles):
            if level == 0.0:
                out.append(self.min)
                continue
            if level == 1.0:
                out.append(self.max)
                continue
            target = level * self.n
            k = int(np.searchsorted(cum, target, side="left"))  # first k with cum[k] >= target
            frac = (target - cum[k - 1]) / (cum[k] - cum[k - 1])
            value = edges[k - 1] + frac * self.bin_width
            out.append(float(np.clip(value, self.min, self.max)))
        return out

    def to_row(self, measurement: str, unit: str, n_quantiles: int) -> dict:
        if self.n == 0:
            return {"measurement": measurement, "unit": unit, "n_observations": 0}
        mean = self.total / self.n
        var = max(self.total_sq / self.n - mean**2, 0.0)
        return {
            "measurement": measurement,
            "unit": unit,
            "n_observations": self.n,
            "mean": mean,
            "std": float(np.sqrt(var)),
            "quantiles": self.quantiles(n_quantiles),
        }


def fit_age_stats(schema_dfs: list[pl.LazyFrame], n_quantiles: int) -> dict:
    """Computes the age statistics row from a collection of (training) subject schemas.

    Examples:
        >>> from datetime import datetime
        >>> shard_0 = pl.DataFrame({
        ...     "time": [[datetime(2000, 1, 1), datetime(2010, 1, 1), datetime(2020, 1, 1)]],
        ...     "birth_time": [datetime(2000, 1, 1)],
        ... }).lazy()
        >>> shard_1 = pl.DataFrame({
        ...     "time": [[datetime(1990, 1, 1), datetime(2020, 1, 1)], [datetime(2001, 1, 1)]],
        ...     "birth_time": [datetime(1990, 1, 1), None],
        ... }).lazy()
        >>> row = fit_age_stats([shard_0, shard_1], n_quantiles=3)
        >>> row["n_observations"], round(row["mean"], 2), round(row["std"], 2)
        (3, 20.0, 8.16)
        >>> [round(q, 2) for q in row["quantiles"]]
        [10.0, 20.0, 30.0]

        With no birth events at all, only the count (0) is recorded:

        >>> fit_age_stats([shard_1.with_columns(pl.lit(None).alias("birth_time"))], n_quantiles=3)
        {'measurement': 'age', 'unit': 'years', 'n_observations': 0}
    """
    stats = _RunningStats(AGE_HIST_MIN_YEARS, AGE_HIST_MAX_YEARS, AGE_HIST_BIN_WIDTH_YEARS)
    for schema_df in schema_dfs:
        stats.update(non_birth_event_ages(schema_df).collect()["age"].to_numpy())
    return stats.to_row("age", "years", n_quantiles)


@Stage.register(is_metadata=True, example_class=MTDStageExample)
def main(cfg: DictConfig):
    """Fits training-split statistics of time-derived measurements from the tokenized subject schemas.

    Reads `schemas/train/**/*.parquet` from the stage's data input directory (the tokenization stage's
    output), writes `time_derived_measurement_stats.parquet` to the metadata output directory, and passes
    every file in the metadata input directory through unchanged.
    """
    data_input_dir = Path(cfg.stage_cfg.data_input_dir)
    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    output_dir = Path(cfg.stage_cfg.reducer_output_dir)
    n_quantiles = int(cfg.stage_cfg.get("n_quantiles", 1001))
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be at least 2; got {n_quantiles}.")

    schema_fps = sorted((data_input_dir / "schemas" / train_split).rglob("*.parquet"))
    if not schema_fps:
        raise FileNotFoundError(f"No training schema shards found under {data_input_dir / 'schemas'}.")
    logger.info(f"Fitting time-derived measurement stats over {len(schema_fps)} training schema shards.")

    schema_dfs = [pl.scan_parquet(fp).select("time", "birth_time") for fp in schema_fps]
    rows = [fit_age_stats(schema_dfs, n_quantiles)]
    if rows[0]["n_observations"] == 0:
        logger.warning(
            "No training subject has a birth event, so age statistics are empty; age-based dataset "
            "features will be unavailable. Check the tokenization stage's `birth_code_regex`."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata_input_dir.is_dir() and metadata_input_dir.resolve() != output_dir.resolve():
        for fp in metadata_input_dir.iterdir():
            if fp.is_file():
                shutil.copy2(fp, output_dir / fp.name)

    pl.DataFrame(rows, schema=STATS_SCHEMA).write_parquet(output_dir / STATS_FN, use_pyarrow=True)
    logger.info(f"Wrote time-derived measurement stats to {output_dir / STATS_FN}.")
