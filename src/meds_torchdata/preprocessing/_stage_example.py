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
    `JointNestedRaggedTensorDict.equals(equal_nan=True)` for NRT.
    """

    # Redeclare as Path — `want_data` here is a yaml spec file, not a parsed MEDSDataset.
    want_data: Path | None = None

    @classmethod
    def is_example_dir(cls, path: Path) -> bool:
        return (path / "out_data.yaml").is_file()

    @classmethod
    def from_dir(cls, stage_name, scenario_name, example_dir, **schema_updates):
        """Load an `MTDStageExample` from a scenario directory on disk.

        Expects `out_data.yaml` (required), optionally `in.yaml` and `cfg.yaml`. A
        non-mapping `cfg.yaml` (list, scalar, etc.) is a user-error — the parent
        `StageExample` treats `stage_cfg` as a dict, so surface the type mismatch
        here with a clear message rather than letting the downstream `.items()` call
        raise a generic `AttributeError`.

        Examples:
            >>> from yaml_to_disk import yaml_disk
            >>> with yaml_disk({
            ...     "out_data.yaml": {"data/x.parquet": {"a": [1, 2]}},
            ...     "cfg.yaml": ["not", "a", "mapping"],
            ... }) as example_dir:
            ...     MTDStageExample.from_dir("demo", "default", example_dir)
            Traceback (most recent call last):
                ...
            TypeError: ...cfg.yaml must contain a YAML mapping; got list
        """
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
        # `is_example_dir` already required `out_data.yaml`, and `from_dir` always wires
        # `want_data` to that file. meds-torch-data has no metadata-only stages, so the
        # `want_data is None` path inherited from the parent `StageExample` contract is
        # unreachable here — don't guard for it.
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
    """Per-file comparison helper for `check_outputs`.

    Dispatches on `expected_fp.suffix`. `.parquet` → `polars.testing.assert_frame_equal`
    (wrapped in an AssertionError that names the rel-path and shows both frames);
    `.nrt` → `JointNestedRaggedTensorDict.equals(equal_nan=True)`; anything else
    raises — the caller is responsible for filtering to allowed suffixes.

    Examples:
        Matching parquets pass silently:

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     pl.DataFrame({"a": [1, 2]}).write_parquet(d / "a.parquet")
        ...     pl.DataFrame({"a": [1, 2]}).write_parquet(d / "b.parquet")
        ...     _compare(d / "a.parquet", d / "b.parquet", Path("x.parquet"), {})

        Differing parquets raise with the rel-path in the message:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     pl.DataFrame({"a": [1, 2, 3]}).write_parquet(d / "a.parquet")
        ...     pl.DataFrame({"a": [1, 2, 999]}).write_parquet(d / "b.parquet")
        ...     _compare(d / "a.parquet", d / "b.parquet", Path("shard/0.parquet"), {})
        Traceback (most recent call last):
            ...
        AssertionError: Parquet shard/0.parquet differs...

        NRT mismatches use `equals(equal_nan=True)` and raise similarly:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     JointNestedRaggedTensorDict({"code": [[1, 2], [3]]}).save(d / "a.nrt")
        ...     JointNestedRaggedTensorDict({"code": [[1, 2], [999]]}).save(d / "b.nrt")
        ...     _compare(d / "a.nrt", d / "b.nrt", Path("data/0.nrt"), {})
        Traceback (most recent call last):
            ...
        AssertionError: NRT data/0.nrt differs...

        Any other suffix raises — the `check_outputs` caller already filters to the
        allowed suffixes, so reaching this branch is a programming error:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "x.json"
        ...     _ = fp.write_text("{}")
        ...     _compare(fp, fp, Path("x.json"), {})
        Traceback (most recent call last):
            ...
        AssertionError: Unsupported output suffix '.json' for x.json.
        MTDStageExample handles '.parquet' and '.nrt' only.
    """
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
            got = JointNestedRaggedTensorDict(tensors_fp=actual_fp)
            want = JointNestedRaggedTensorDict(tensors_fp=expected_fp)
            assert want.equals(got, equal_nan=True), f"NRT {rel} differs.\nGot:\n{got}\nWant:\n{want}"
        case _:
            raise AssertionError(
                f"Unsupported output suffix '{expected_fp.suffix}' for {rel}. "
                f"MTDStageExample handles '.parquet' and '.nrt' only."
            )
