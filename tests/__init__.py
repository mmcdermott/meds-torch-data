import polars as pl

MEDS_PL_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
}

DEFAULT_CSV_TS_FORMAT = "%m/%d/%Y, %H:%M:%S"
