"""A `StageExample` subclass for meds-torch-data's preprocessing stages.

The built-in `MEDS_transforms.stages.examples.StageExample` validates stage outputs under
`data/*.parquet` (as MEDS-format shards) and `metadata/codes.parquet` only. Neither of our
stages fits that mold:

- **tokenization** writes per-subject schema parquets (`schemas/<shard>.parquet`) and event-seq
  parquets (`event_seqs/<shard>.parquet`). The schema parquet carries list-of-list columns
  (static_code, time, measurements_per_event, …) that are not MEDS-format.
- **tensorization** writes `.nrt` (JointNestedRaggedTensorDict) files under `data/<shard>.nrt`.

One shared subclass is enough. Expected outputs are declared in a yaml_to_disk-style
`out_data.yaml` mapping shard-relative paths to either a column-map (for `.parquet` paths)
or a tensor-map (for `.nrt` paths). `check_outputs` dispatches on the suffix.

Modeled on the upstream `JsonOutputStageExample` in `simple_example_pkg` — see
https://github.com/mmcdermott/MEDS_transforms/blob/main/example/simple_example_pkg/src/simple_example_pkg/export_code_summary/export_code_summary.py
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


@dataclass
class MTDStageExample(StageExample):
    """`StageExample` subclass that validates `.parquet` and `.nrt` outputs.

    The expected-output spec (`out_data.yaml`) is a flat mapping from shard-relative path to
    structured data:

        # out_data.yaml
        schemas/train/0.parquet:
          subject_id: [239684, 1195293]
          static_code: [[7, 9], [6, 9]]
          ...
        data/train/0.nrt:
          code: [[[5], [1, 10, 11], ...], ...]
          time_delta_days: [[.nan, ...], ...]
          numeric_value: [[[.nan], ...], ...]

    Paths ending in `.parquet` are compared against polars DataFrames (via
    `polars.testing.assert_frame_equal`). Paths ending in `.nrt` are compared against
    `JointNestedRaggedTensorDict` tensors (via `np.array_equal(equal_nan=True)` per key).
    """

    # Redeclare as Path — `want_data` here is a yaml spec file, not a parsed MEDSDataset.
    # (The parent expects a MEDSDataset; we repurpose the slot the same way the upstream
    # `JsonOutputStageExample` does.)
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

        spec = safe_load(self.want_data.read_text())
        if not isinstance(spec, dict):
            raise AssertionError(
                f"{self.want_data} must contain a top-level mapping from shard-relative paths "
                f"to expected output contents; got {type(spec).__name__}."
            )
        for rel_path, contents in spec.items():
            # Paths in `out_data.yaml` are relative to the cohort root in standalone mode
            # (e.g. `data/schemas/train/0.parquet`). Under `pipeline_tester`'s intermediate-
            # stage validation, `output_dir` is already resolved to the stage's per-stage
            # subdir (`${cohort}/<stage_name>/`), so the leading `data/` segment has been
            # absorbed. Strip it when `is_resolved_dir=True`.
            effective_rel = rel_path
            if is_resolved_dir and effective_rel.startswith("data/"):
                effective_rel = effective_rel[len("data/") :]
            actual_fp = output_dir / effective_rel

            if not actual_fp.is_file():
                existing = sorted(p.relative_to(output_dir) for p in output_dir.rglob("*") if p.is_file())
                raise AssertionError(
                    f"Expected output file {effective_rel} not found in {output_dir}. "
                    f"Existing files: {existing}"
                )

            suffix = Path(rel_path).suffix
            match suffix:
                case ".parquet":
                    _check_parquet(rel_path, actual_fp, contents, self.df_check_kwargs)
                case ".nrt":
                    _check_nrt(rel_path, actual_fp, contents)
                case _:
                    raise AssertionError(
                        f"Unsupported output suffix '{suffix}' for {rel_path}. "
                        f"MTDStageExample handles '.parquet' and '.nrt' only."
                    )


def _check_parquet(rel_path: str, actual_fp: Path, want_cols: dict, df_check_kwargs: dict) -> None:
    got = pl.read_parquet(actual_fp)
    want = pl.DataFrame(want_cols)
    # YAML doesn't natively express polars dtypes (u32 vs i64, f32 vs f64, etc.), so leave
    # `check_dtypes` off by default and let callers override via `df_check_kwargs` when they
    # want strict dtype validation. Value comparison + structural shape is the contract.
    kwargs = {"check_column_order": False, "check_dtypes": False, **df_check_kwargs}
    try:
        assert_frame_equal(got, want, **kwargs)
    except AssertionError as e:
        raise AssertionError(
            f"Parquet output {rel_path} differs from expected.\nGot:\n{got}\nWant:\n{want}"
        ) from e


def _check_nrt(rel_path: str, actual_fp: Path, want_tensors: dict) -> None:
    got_nrt = JointNestedRaggedTensorDict(tensors_fp=actual_fp)
    want_nrt = JointNestedRaggedTensorDict(want_tensors)

    got_keys, want_keys = set(got_nrt.tensors), set(want_nrt.tensors)
    assert got_keys == want_keys, (
        f"NRT {rel_path} tensor keys differ. Want {sorted(want_keys)}, got {sorted(got_keys)}."
    )

    for k, want in want_nrt.tensors.items():
        got = got_nrt.tensors[k]
        if isinstance(want, list):
            assert len(want) == len(got), (
                f"NRT {rel_path} tensor '{k}' has {len(got)} elements, want {len(want)}."
            )
            for i, (w, g) in enumerate(zip(want, got, strict=True)):
                assert np.array_equal(w, g, equal_nan=True), (
                    f"NRT {rel_path} tensor '{k}[{i}]' differs.\nGot: {g}\nWant: {w}"
                )
        else:
            assert np.array_equal(want, got, equal_nan=True), (
                f"NRT {rel_path} tensor '{k}' differs.\nGot: {got}\nWant: {want}"
            )
