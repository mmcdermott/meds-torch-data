"""A `StageExample` subclass for meds-torch-data's preprocessing stages.

The built-in `MEDS_transforms.stages.examples.StageExample` validates stage outputs under
`data/*.parquet` (as MEDS-format shards) and `metadata/codes.parquet` only. Neither of our
stages fits that mold:

- **tokenization** writes per-subject schema parquets (`schemas/<shard>.parquet`) and event-seq
  parquets (`event_seqs/<shard>.parquet`). The schema parquet carries list-of-list columns
  (static_code, time, measurements_per_event, …) that are not MEDS-format.
- **tensorization** writes `.nrt` (JointNestedRaggedTensorDict) files under `data/<shard>.nrt`.

Expected outputs are declared in an `out_data.yaml` mapping shard-relative paths to structured
file contents. `check_outputs` materializes that YAML to a temp dir via `yaml_to_disk` (which
handles `.parquet` natively and `.nrt` via the `NRTFile` plugin in `_nrt_file.py`), then
walks the materialized tree and file-by-file compares each expected file to the stage's
actual output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from MEDS_transforms.stages.examples import StageExample
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from polars.testing import assert_frame_equal
from yaml import safe_load
from yaml_to_disk import yaml_disk

_ALLOWED_SUFFIXES = frozenset({".parquet", ".nrt"})


@dataclass
class MTDStageExample(StageExample):
    """`StageExample` subclass that validates `.parquet` and `.nrt` outputs via yaml_to_disk.

    `yaml_to_disk` materializes the `out_data.yaml` spec into a real directory of
    `.parquet` / `.nrt` files (the former natively, the latter via the `NRTFile` plugin
    registered in `pyproject.toml`). `check_outputs` then compares each materialized
    expected file to the stage's actual output — Polars `assert_frame_equal` for parquet,
    per-tensor `np.array_equal(equal_nan=True)` for NRT.
    """

    # Redeclare as Path — `want_data` here is a yaml spec file, not a parsed MEDSDataset.
    want_data: Path | None = None

    @classmethod
    def is_example_dir(cls, path: Path) -> bool:
        return (path / "out_data.yaml").is_file()

    @classmethod
    def from_dir(cls, stage_name, scenario_name, example_dir, **schema_updates):
        want_data_fp = example_dir / "out_data.yaml"
        in_fp = example_dir / "in.yaml"
        stage_cfg_fp = example_dir / "cfg.yaml"

        stage_cfg = {}
        if stage_cfg_fp.is_file():
            stage_cfg = safe_load(stage_cfg_fp.read_text()) or {}
            if not isinstance(stage_cfg, dict):
                raise TypeError(f"{stage_cfg_fp} must contain a YAML mapping; got {type(stage_cfg).__name__}")

        return cls(
            stage_name=stage_name,
            scenario_name=scenario_name,
            want_data=want_data_fp if want_data_fp.is_file() else None,
            in_data=in_fp if in_fp.is_file() else None,
            stage_cfg=stage_cfg,
        )

    def check_outputs(self, output_dir: Path, is_resolved_dir: bool = False) -> None:
        if self.want_data is None:
            return

        with yaml_disk(self.want_data) as expected_root:
            # Under pipeline_tester's per-stage validation, `output_dir` is already resolved
            # to `${cohort}/<stage_name>/`, so the leading `data/` segment in expected paths
            # has been absorbed by the caller.
            def _canonical(rel: Path) -> Path:
                if is_resolved_dir and rel.parts and rel.parts[0] == "data":
                    return Path(*rel.parts[1:])
                return rel

            expected = {
                _canonical(fp.relative_to(expected_root)): fp
                for fp in expected_root.rglob("*")
                if fp.is_file() and fp.suffix in _ALLOWED_SUFFIXES
            }
            # Scan only the top-level dirs present in `expected` so we don't pick up
            # unrelated artifacts (upstream MEDS metadata, hydra logs, etc.).
            actual = {
                fp.relative_to(output_dir): fp
                for top in {rel.parts[0] for rel in expected}
                for fp in (output_dir / top).rglob("*")
                if fp.is_file() and fp.suffix in _ALLOWED_SUFFIXES
            }

            assert expected.keys() == actual.keys(), (
                f"Output file set mismatch in {output_dir}.\n"
                f"Missing:    {sorted(expected.keys() - actual.keys())}\n"
                f"Unexpected: {sorted(actual.keys() - expected.keys())}"
            )
            for rel in sorted(expected):
                _compare(expected[rel], actual[rel], rel, self.df_check_kwargs)


def _compare(expected_fp: Path, actual_fp: Path, rel: Path, df_check_kwargs: dict) -> None:
    match expected_fp.suffix:
        case ".parquet":
            # YAML can't express polars dtypes (u32 vs i64, f32 vs f64, etc.) and pyarrow's
            # type inference diverges from polars' for nested lists, so leave `check_dtypes`
            # off by default and let callers override via `df_check_kwargs`.
            kwargs = {"check_column_order": False, "check_dtypes": False, **df_check_kwargs}
            got = pl.read_parquet(actual_fp)
            want = pl.read_parquet(expected_fp)
            try:
                assert_frame_equal(got, want, **kwargs)
            except AssertionError as e:
                raise AssertionError(f"Parquet {rel} differs.\nGot:\n{got}\nWant:\n{want}") from e
        case ".nrt":
            # Can't use `want == got` — `JointNestedRaggedTensorDict.__eq__` compares via
            # `np.array_equal` without `equal_nan=True`, so any NaN (e.g., the leading
            # `time_delta`) makes the comparison return False. See upstream issue:
            # https://github.com/mmcdermott/nested_ragged_tensors/issues/63
            got = JointNestedRaggedTensorDict(tensors_fp=actual_fp)
            want = JointNestedRaggedTensorDict(tensors_fp=expected_fp)
            assert set(got.tensors) == set(want.tensors), (
                f"NRT {rel} tensor keys differ. Want {sorted(want.tensors)}, got {sorted(got.tensors)}."
            )
            for k, want_v in want.tensors.items():
                got_v = got.tensors[k]
                if isinstance(want_v, list):
                    for i, (w, g) in enumerate(zip(want_v, got_v, strict=True)):
                        assert np.array_equal(w, g, equal_nan=True), (
                            f"NRT {rel} tensor '{k}[{i}]' differs.\nGot: {g}\nWant: {w}"
                        )
                else:
                    assert np.array_equal(want_v, got_v, equal_nan=True), (
                        f"NRT {rel} tensor '{k}' differs.\nGot: {got_v}\nWant: {want_v}"
                    )
        case _:
            raise AssertionError(
                f"Unsupported output suffix '{expected_fp.suffix}' for {rel}. "
                f"MTDStageExample handles '.parquet' and '.nrt' only."
            )
