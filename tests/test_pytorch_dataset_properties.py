from bisect import bisect_right
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from meds import DataSchema, LabelSchema

from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.types import BatchMode


def _schema_and_labels():
    """Strategy generating a schema DataFrame and a corresponding label DataFrame."""

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
        schema_times = []
        for _ in subject_ids:
            n_times = draw(st.integers(min_value=1, max_value=5))
            times = draw(
                st.lists(st.datetimes(min_value=start, max_value=end), min_size=n_times, max_size=n_times)
            )
            schema_times.append(sorted(times))
        schema_df = pl.DataFrame(
            {DataSchema.subject_id_name: subject_ids, DataSchema.time_name: schema_times}
        )

        n_labels = draw(st.integers(min_value=1, max_value=6))
        label_rows = []
        for _ in range(n_labels):
            subj = draw(st.one_of(st.sampled_from(subject_ids), st.integers(min_value=50, max_value=60)))
            pred_time = draw(
                st.datetimes(min_value=start - timedelta(days=1), max_value=end + timedelta(days=1))
            )
            value = draw(st.booleans())
            label_rows.append(
                {
                    DataSchema.subject_id_name: subj,
                    LabelSchema.prediction_time_name: pred_time,
                    LabelSchema.boolean_value_name: value,
                }
            )
        label_df = pl.DataFrame(label_rows)
        return schema_df, label_df

    return _strategy()


@given(_schema_and_labels())
@settings(max_examples=25, deadline=None)
def test_get_task_seq_bounds_and_labels_property(data):
    schema_df, label_df = data

    result = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)

    # Drop labels for subjects not in schema_df
    label_subset = label_df.filter(
        pl.col(DataSchema.subject_id_name).is_in(schema_df[DataSchema.subject_id_name].implode())
    )

    expected_rows = []
    schema_map = {
        row[0]: row[1]
        for row in schema_df.select(DataSchema.subject_id_name, DataSchema.time_name).iter_rows()
    }
    for row in label_subset.iter_rows(named=True):
        times = schema_map[row[DataSchema.subject_id_name]]
        idx = bisect_right(times, row[LabelSchema.prediction_time_name])
        expected_rows.append(
            {
                DataSchema.subject_id_name: row[DataSchema.subject_id_name],
                MEDSPytorchDataset.END_IDX: idx,
                LabelSchema.prediction_time_name: row[LabelSchema.prediction_time_name],
                LabelSchema.boolean_value_name: row[LabelSchema.boolean_value_name],
            }
        )

    expected = pl.DataFrame(expected_rows, schema=result.schema)

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
    schema_df, label_df = data
    result = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)

    schema_map = {
        row[0]: row[1]
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
