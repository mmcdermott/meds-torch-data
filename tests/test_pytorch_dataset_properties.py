from datetime import datetime, timedelta

import numpy as np
import polars as pl
from hypothesis import example, given, settings
from hypothesis import strategies as st
from meds import DataSchema, LabelSchema

from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.types import BatchMode


def _schema_and_labels(
    *, with_label: bool = True, allow_empty_times: bool = True, allow_unsorted: bool = True
):
    """Strategy generating a (schema_df, label_df) pair for `get_task_seq_bounds_and_labels`.

    Explores the full contract surface:

    - `allow_empty_times=True` occasionally produces subjects with `[]` time lists,
      exercising the static-only-subject edge case (see issue #92 review).
    - `allow_unsorted=True` occasionally produces non-chronological schema time
      lists; the implementation must not silently produce wrong indices when the
      upstream sorted-time invariant is violated.
    - Labels sometimes reference subjects absent from `schema_df` (drawn from a
      disjoint integer range) to exercise the inner-join drop path.
    - Prediction times can sit before, on, or after the subject's event times.

    Args:
        with_label: If True, include a `boolean_value` column on the label DF.
        allow_empty_times: If True, permit subjects with zero events in schema_df.
        allow_unsorted: If True, permit schema time lists in non-chronological order.
    """

    @st.composite
    def _strategy(draw):
        n_subjects = draw(st.integers(min_value=1, max_value=4))
        subject_ids = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=n_subjects, max_size=n_subjects, unique=True
            )
        )
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        min_times = 0 if allow_empty_times else 1
        schema_times = []
        for _ in subject_ids:
            n_times = draw(st.integers(min_value=min_times, max_value=5))
            times = draw(
                st.lists(st.datetimes(min_value=start, max_value=end), min_size=n_times, max_size=n_times)
            )
            # Sorted is the upstream-MEDS invariant; permitting unsorted here catches
            # the sort-before-index bug class flagged during review of #92.
            if allow_unsorted and draw(st.booleans()):
                schema_times.append(times)
            else:
                schema_times.append(sorted(times))
        # Construct schema_df with explicit dtypes so empty-list subjects don't
        # degrade the time column to List(Null) under polars' type inference —
        # real schema_df comes from parquet and always carries List(Datetime).
        schema_df = pl.DataFrame(
            {DataSchema.subject_id_name: subject_ids, DataSchema.time_name: schema_times},
            schema={
                DataSchema.subject_id_name: pl.Int64,
                DataSchema.time_name: pl.List(pl.Datetime("us")),
            },
        )

        n_labels = draw(st.integers(min_value=1, max_value=6))
        label_rows = []
        for _ in range(n_labels):
            subj = draw(st.one_of(st.sampled_from(subject_ids), st.integers(min_value=50, max_value=60)))
            pred_time = draw(
                st.datetimes(min_value=start - timedelta(days=1), max_value=end + timedelta(days=1))
            )
            row = {
                DataSchema.subject_id_name: subj,
                LabelSchema.prediction_time_name: pred_time,
            }
            if with_label:
                row[LabelSchema.boolean_value_name] = draw(st.booleans())
            label_rows.append(row)
        label_schema = {
            DataSchema.subject_id_name: pl.Int64,
            LabelSchema.prediction_time_name: pl.Datetime("us"),
        }
        if with_label:
            label_schema[LabelSchema.boolean_value_name] = pl.Boolean
        label_df = pl.DataFrame(label_rows, schema=label_schema)
        return schema_df, label_df

    return _strategy()


def _reference_end_idx(times: list[datetime], prediction_time: datetime) -> int:
    """Order-independent reference: count of events with `time <= prediction_time`.

    Used instead of `bisect_right` because the strategy can produce unsorted time
    lists; the implementation's contract is a *count*, not a sorted-position lookup.
    """
    return sum(1 for t in times if t <= prediction_time)


@given(_schema_and_labels())
@settings(max_examples=50, deadline=None)
# Explicit anchors for the edge cases flagged during #92 review — keep coverage of
# these exact cases even if the random strategy drifts.
@example(
    data=(
        # Subject 5 has an empty time list; subject 1's times are deliberately unsorted.
        pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 2, 5],
                DataSchema.time_name: [
                    [datetime(2020, 1, 3), datetime(2020, 1, 1), datetime(2020, 1, 2)],
                    [datetime(2020, 6, 1), datetime(2020, 6, 2)],
                    [],
                ],
            }
        ),
        pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 5, 9],
                LabelSchema.prediction_time_name: [
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 1),  # subject 9 absent — dropped
                ],
                LabelSchema.boolean_value_name: [True, False, True],
            }
        ),
    ),
)
def test_get_task_seq_bounds_and_labels_property(data):
    schema_df, label_df = data

    result = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)

    # Order-independent reference (count of events with time <= prediction_time),
    # robust to unsorted schema time lists and empty lists.
    schema_map = {
        row[0]: row[1] or []
        for row in schema_df.select(DataSchema.subject_id_name, DataSchema.time_name).iter_rows()
    }
    label_subset = label_df.filter(pl.col(DataSchema.subject_id_name).is_in(list(schema_map)))

    has_label = LabelSchema.boolean_value_name in label_df.columns
    expected_rows = []
    for row in label_subset.iter_rows(named=True):
        times = schema_map[row[DataSchema.subject_id_name]]
        idx = _reference_end_idx(times, row[LabelSchema.prediction_time_name])
        expected = {
            DataSchema.subject_id_name: row[DataSchema.subject_id_name],
            MEDSPytorchDataset.END_IDX: idx,
            LabelSchema.prediction_time_name: row[LabelSchema.prediction_time_name],
        }
        if has_label:
            expected[LabelSchema.boolean_value_name] = row[LabelSchema.boolean_value_name]
        expected_rows.append(expected)

    expected = pl.DataFrame(expected_rows, schema=result.schema)
    assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)


@given(_schema_and_labels(with_label=False))
@settings(max_examples=25, deadline=None)
def test_get_task_seq_bounds_and_labels_unlabeled(data):
    """Unlabeled index-mode callers: `label_df` without `boolean_value` column."""
    schema_df, label_df = data

    result = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)
    assert LabelSchema.boolean_value_name not in result.columns

    schema_map = {
        row[0]: row[1] or []
        for row in schema_df.select(DataSchema.subject_id_name, DataSchema.time_name).iter_rows()
    }
    label_subset = label_df.filter(pl.col(DataSchema.subject_id_name).is_in(list(schema_map)))
    expected = pl.DataFrame(
        [
            {
                DataSchema.subject_id_name: row[DataSchema.subject_id_name],
                MEDSPytorchDataset.END_IDX: _reference_end_idx(
                    schema_map[row[DataSchema.subject_id_name]],
                    row[LabelSchema.prediction_time_name],
                ),
                LabelSchema.prediction_time_name: row[LabelSchema.prediction_time_name],
            }
            for row in label_subset.iter_rows(named=True)
        ],
        schema=result.schema,
    )
    assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)


@given(st.data())
@settings(max_examples=25, deadline=None)
def test_schema_df_last_observed(sample_dataset_config_with_index, data):
    cfg = sample_dataset_config_with_index
    cfg.include_window_last_observed_in_schema = True
    dataset = MEDSPytorchDataset(cfg, split="train")

    idx = data.draw(st.integers(min_value=0, max_value=len(dataset) - 1))
    subj, end_idx = dataset.index[idx]
    shard, subj_idx = dataset.subj_locations[subj]
    times = dataset.schema_dfs_by_shard[shard][DataSchema.time_name][subj_idx]

    assert 0 < end_idx <= len(times)
    assert dataset.schema_df[dataset.LAST_TIME][idx] == times[end_idx - 1]


@given(_schema_and_labels())
@settings(max_examples=25, deadline=None)
def test_get_task_seq_bounds_and_labels_semantic(data):
    """Output invariants: `end_idx` is bounded and matches `times`-slice semantics.

    The strategy may yield unsorted schema time lists; since the per-subject
    invariants here are about chronological ordering (events-before vs events-after
    the prediction time), sort a local copy before checking. The function's own
    contract is the count of events with `time <= prediction_time`, which is
    order-independent — that's covered by the equivalence test above.
    """
    schema_df, label_df = data
    result = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)

    schema_map = {
        row[0]: sorted(row[1] or [])
        for row in schema_df.select(DataSchema.subject_id_name, DataSchema.time_name).iter_rows()
    }

    for row in result.iter_rows(named=True):
        subj = row[DataSchema.subject_id_name]
        times = schema_map[subj]
        end_idx = row[MEDSPytorchDataset.END_IDX]
        pred_time = row[LabelSchema.prediction_time_name]

        assert 0 <= end_idx <= len(times)
        if end_idx < len(times):
            assert times[end_idx] > pred_time
        else:
            assert end_idx == len(times)
        if end_idx > 0:
            assert times[end_idx - 1] <= pred_time


def test_getitem_consistency(sample_dataset_config_with_index):
    cfg = sample_dataset_config_with_index
    cfg.include_window_last_observed_in_schema = True
    cfg.batch_mode = BatchMode.SEM
    dataset = MEDSPytorchDataset(cfg, split="train")

    for idx in range(len(dataset)):
        item = dataset[idx]
        subj, end_idx = dataset.index[idx]

        shard, subj_idx = dataset.subj_locations[subj]
        times = dataset.schema_dfs_by_shard[shard][DataSchema.time_name][subj_idx]

        dense = item["dynamic"].to_dense()
        deltas = np.asarray(dense["time_delta_days"], dtype=float)
        n_events = deltas.shape[0]

        assert n_events == min(end_idx, dataset.config.max_seq_len)

        start_idx = end_idx - n_events

        time_deltas = [float("nan")]
        for i in range(1, len(times)):
            time_deltas.append((times[i] - times[i - 1]).total_seconds() / (24 * 3600))
        expected_slice = np.asarray(time_deltas[start_idx:end_idx], dtype=float)

        assert deltas.shape[0] == expected_slice.shape[0]
        assert np.allclose(deltas, expected_slice, equal_nan=True)

        prev_time = times[start_idx - 1] if start_idx > 0 else times[0]
        observed_days = np.nansum(np.nan_to_num(deltas))
        expected_days = (times[end_idx - 1] - prev_time).total_seconds() / (24 * 3600)
        assert np.isclose(observed_days, expected_days)
        last_time = prev_time + timedelta(days=float(observed_days))

        assert abs((last_time - times[end_idx - 1]).total_seconds()) < 60
        assert dataset.schema_df[dataset.LAST_TIME][idx] == times[end_idx - 1]

        if dataset.has_task_labels:
            assert item[dataset.LABEL_COL].item() == dataset.schema_df[dataset.LABEL_COL][idx]
