"""Unit tests for the defensive / error-path branches in `MTDStageExample`.

Covers branches that the `stage_example` pytest-plugin fixture doesn't naturally
exercise on our real stage scenarios:

- `cfg.yaml` present in the example dir with non-mapping contents (TypeError).
- `check_outputs` short-circuit when `want_data` is None.
- Per-file diff paths in `_compare`:
  - parquet contents that don't match (wrapped `AssertionError`).
  - unsupported suffix (e.g., `.json`) reaches the `case _:` guard.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from meds_torchdata.preprocessing._stage_example import MTDStageExample, _compare


def test_from_dir_cfg_yaml_must_be_mapping(tmp_path: Path):
    """`cfg.yaml` with top-level list / scalar must raise TypeError."""
    example_dir = tmp_path
    (example_dir / "out_data.yaml").write_text("data/x.parquet:\n  a: [1, 2]\n")
    (example_dir / "cfg.yaml").write_text("- not\n- a\n- mapping\n")

    with pytest.raises(TypeError, match=r"cfg\.yaml must contain a YAML mapping"):
        MTDStageExample.from_dir("demo", "default", example_dir)


def test_check_outputs_noop_when_want_data_none(tmp_path: Path):
    """`check_outputs` returns without reading the output_dir when `want_data` is None.

    The parent `StageExample.__post_init__` requires at least one of
    `want_data` / `want_metadata`, so construct with `want_metadata` set and then
    clear `want_data` post-hoc to reach the `want_data is None` short-circuit.
    """
    example = MTDStageExample(
        stage_name="demo",
        scenario_name="default",
        want_metadata=pl.DataFrame({"code": ["X"], "description": ["x"]}),
    )
    example.want_data = None
    # tmp_path intentionally isn't a real output dir; the function should short-circuit
    # on `want_data is None` before touching it.
    example.check_outputs(tmp_path)


def test_compare_parquet_diff_wraps_assertion(tmp_path: Path):
    """Parquet content mismatch raises a wrapped `AssertionError` naming the rel path."""
    expected_fp = tmp_path / "expected.parquet"
    actual_fp = tmp_path / "actual.parquet"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(expected_fp)
    pl.DataFrame({"a": [1, 2, 999]}).write_parquet(actual_fp)

    with pytest.raises(AssertionError, match=r"Parquet some/rel\.parquet differs"):
        _compare(expected_fp, actual_fp, rel=Path("some/rel.parquet"), df_check_kwargs={})


def test_compare_unsupported_suffix_raises(tmp_path: Path):
    """Only `.parquet` and `.nrt` are handled; anything else hits the `case _:` guard."""
    fp = tmp_path / "x.json"
    fp.write_text("{}")
    with pytest.raises(AssertionError, match=r"Unsupported output suffix '\.json'"):
        _compare(fp, fp, rel=Path("x.json"), df_check_kwargs={})


def test_compare_nrt_detects_diff(tmp_path: Path):
    """Companion coverage: `.nrt` branch surfaces a mismatch via `equals(equal_nan=True)`.

    The real stage-example fixture always runs `_compare` with matching expected/actual,
    so the negative case here locks in the assertion message for future regressions.
    """
    expected_fp = tmp_path / "expected.nrt"
    actual_fp = tmp_path / "actual.nrt"
    JointNestedRaggedTensorDict({"code": [[1, 2], [3]]}).save(expected_fp)
    JointNestedRaggedTensorDict({"code": [[1, 2], [999]]}).save(actual_fp)

    with pytest.raises(AssertionError, match=r"NRT shard/0\.nrt differs"):
        _compare(expected_fp, actual_fp, rel=Path("shard/0.nrt"), df_check_kwargs={})
