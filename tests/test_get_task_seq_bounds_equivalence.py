"""Regression tests for `MEDSPytorchDataset.get_task_seq_bounds_and_labels`.

Validates the polars-native implementation against a direct Python-loop reference
across a set of edge cases flagged during review of PR #97:

- prediction_time before the subject's first event
- prediction_time exactly on an event time
- prediction_time after the subject's last event
- label_df without the `boolean_value` column (unlabeled indexing)
- subject with an empty time list in schema_df
- label subject absent from schema_df (inner-join drop)
- schema_df with time list in non-sorted order (historically always sorted upstream,
  but the implementation must not silently produce wrong indices when it isn't)

The reference implementation is a plain Python loop computing
`end_idx = count(events with time <= prediction_time)` per label.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from meds_torchdata.pytorch_dataset import MEDSPytorchDataset


def _reference(label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
    """Straight-line Python reference for `get_task_seq_bounds_and_labels`.

    Computes `end_idx = count(events with time <= prediction_time)` per label row, dropping labels whose
    subject_id isn't in schema_df. Preserves the ordering of retained rows from the input label_df.
    """
    has_label = "boolean_value" in label_df.columns
    events_by_subj = {r["subject_id"]: list(r["time"] or []) for r in schema_df.to_dicts()}

    out_rows = []
    for row in label_df.to_dicts():
        sid = row["subject_id"]
        if sid not in events_by_subj:
            continue
        pt = row["prediction_time"]
        end_idx = sum(1 for t in events_by_subj[sid] if t <= pt)
        out_row = {
            "subject_id": sid,
            "end_event_index": end_idx,
            "prediction_time": pt,
        }
        if has_label:
            out_row["boolean_value"] = row["boolean_value"]
        out_rows.append(out_row)

    schema = {
        "subject_id": pl.Int64,
        "end_event_index": pl.UInt32,
        "prediction_time": pl.Datetime("us"),
    }
    if has_label:
        schema["boolean_value"] = pl.Boolean
    return pl.DataFrame(out_rows, schema=schema)


@pytest.fixture
def schema_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 5],
            "time": [
                [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
                [datetime(2020, 6, 1), datetime(2020, 6, 2)],
                [datetime(2020, 1, 1)],
                # Subject 5 has no dynamic events (static-only) — schema exposes an
                # empty time list for such subjects. The function must not crash.
                [],
            ],
        }
    )


@pytest.mark.parametrize(
    "label_rows, label_value",
    [
        # Before first event — every subject's earliest-allowed window is 0.
        pytest.param(
            [(1, datetime(2019, 12, 31)), (2, datetime(2020, 5, 1))],
            [True, False],
            id="before_first",
        ),
        # Exactly on an event time — side="right" semantics include the matched event.
        pytest.param(
            [(1, datetime(2020, 1, 1)), (1, datetime(2020, 1, 2)), (1, datetime(2020, 1, 3))],
            [True, False, True],
            id="exactly_on_event",
        ),
        # After last event — all events included.
        pytest.param(
            [(1, datetime(2099, 1, 1)), (2, datetime(2099, 1, 1))],
            [False, True],
            id="after_last",
        ),
        # Mixed: a bit of everything per subject.
        pytest.param(
            [
                (1, datetime(2019, 12, 31)),
                (1, datetime(2020, 1, 2)),
                (2, datetime(2020, 6, 1, 12)),  # between events
                (3, datetime(2020, 1, 1)),
                (3, datetime(2099, 1, 1)),
            ],
            [True, False, True, False, True],
            id="mixed",
        ),
        # Subject 4 is not in schema_df — should be dropped.
        pytest.param(
            [(1, datetime(2020, 1, 2)), (4, datetime(2020, 1, 1)), (2, datetime(2020, 6, 2))],
            [True, False, True],
            id="drops_absent_subject",
        ),
        # Subject 5 has empty time list — any label falls in "before first" territory
        # and gets end_idx = 0. The row is still emitted (subject IS in schema_df).
        pytest.param(
            [(5, datetime(2020, 1, 1)), (1, datetime(2020, 1, 1))],
            [True, False],
            id="empty_event_list",
        ),
    ],
)
def test_matches_reference(schema_df, label_rows, label_value):
    label_df = pl.DataFrame(
        {
            "subject_id": [r[0] for r in label_rows],
            "prediction_time": [r[1] for r in label_rows],
            "boolean_value": label_value,
        }
    )
    got = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)
    want = _reference(label_df, schema_df)
    assert got.equals(want), f"Mismatch.\nGot:\n{got}\nWant:\n{want}"


def test_unlabeled(schema_df):
    """`boolean_value` column omission — index-mode callers don't carry labels."""
    label_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [datetime(2020, 1, 2), datetime(2020, 6, 1, 12), datetime(2020, 1, 1)],
        }
    )
    got = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)
    want = _reference(label_df, schema_df)
    assert "boolean_value" not in got.columns, "boolean_value should not be synthesized"
    assert got.equals(want)


def test_unsorted_schema_time_list(schema_df):
    """Guard against the sort-before-index bug flagged by Copilot on PR #97.

    Upstream tokenization produces sorted per-subject time lists, but the function
    should not silently produce wrong `end_event_index` values if that invariant is
    violated — the index is now computed after sorting inside the function.
    """
    # Shuffle subject 1's time list in-place; semantic content is unchanged.
    unsorted_schema = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 5],
            "time": [
                [datetime(2020, 1, 3), datetime(2020, 1, 1), datetime(2020, 1, 2)],  # shuffled
                [datetime(2020, 6, 2), datetime(2020, 6, 1)],  # shuffled
                [datetime(2020, 1, 1)],
                [],
            ],
        }
    )
    label_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2],
            "prediction_time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 6, 1, 12),
            ],
            "boolean_value": [True, False, True, False],
        }
    )
    got = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, unsorted_schema)
    # Reference uses the unsorted schema but semantically computes "how many
    # events <= prediction_time" which doesn't depend on list order.
    want = _reference(label_df, unsorted_schema)
    assert got.equals(want), f"Got:\n{got}\nWant:\n{want}"
