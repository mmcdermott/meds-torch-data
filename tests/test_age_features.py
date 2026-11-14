"""Tests for the time-derived age features (`include_age`, `age_token_mode`, `age_bin_edges_years`).

Expected ages are computed independently from the raw MEDS data (the `DOB` row and the unique event times
per subject), not from the tensorized schema the dataset itself reads.
"""

import dataclasses

import numpy as np
import polars as pl
import pytest
import torch
from meds_testing_helpers.dataset import MEDSDataset

from meds_torchdata import MEDSPytorchDataset

DAYS_PER_YEAR = 365.25


@pytest.fixture(scope="module")
def raw_event_ages(simple_static_MEDS) -> dict[int, np.ndarray]:
    """Per-subject age (years) at each unique dynamic event time, from the raw MEDS shards."""
    D = MEDSDataset(root_dir=simple_static_MEDS)
    df = pl.concat([pl.from_arrow(shard) for shard in D.data_shards.values()])
    dob = df.filter(pl.col("code") == "DOB").select("subject_id", pl.col("time").alias("dob"))
    events = (
        df.filter(pl.col("time").is_not_null())
        .select("subject_id", "time")
        .unique()
        .join(dob, on="subject_id")
        .with_columns(
            ((pl.col("time") - pl.col("dob")).dt.total_seconds() / 86400 / DAYS_PER_YEAR).alias("age")
        )
        .sort("subject_id", "time")
    )
    return {sid: grp["age"].to_numpy() for (sid,), grp in events.group_by("subject_id", maintain_order=True)}


def _age_cfg(sample_dataset_config, **kwargs):
    defaults = {"birth_codes": ["DOB"], "seq_sampling_strategy": "to_end", "max_seq_len": 6}
    return dataclasses.replace(sample_dataset_config, **{**defaults, **kwargs})


def _full_batch(pyd: MEDSPytorchDataset):
    return pyd.collate([pyd[i] for i in range(len(pyd))])


def _expected_sm_window_ages(pyd: MEDSPytorchDataset, idx: int, raw_ages: dict) -> np.ndarray:
    """Age at each measurement of the sampled (TO_END) window for dataset element `idx` in SM mode."""
    subject_id, end_idx = pyd.index[idx]
    dynamic, _ = pyd.load_subject_data(subject_id, 0, end_idx)
    meas_per_event = np.diff(dynamic.tensors["dim1/bounds"], prepend=0)
    per_meas = np.repeat(raw_ages[subject_id][:end_idx], meas_per_event)
    n_window = len(pyd[idx]["dynamic"]) - pyd[idx].get("n_static_seq_els", 0)
    return per_meas[len(per_meas) - n_window :]


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_sm_per_position_age_matches_raw_data(sample_dataset_config, raw_event_ages, padding_side):
    cfg = _age_cfg(sample_dataset_config, include_age=True, padding_side=padding_side)
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)

    assert batch.age_years.shape == batch.code.shape
    assert batch.age_mask.dtype == torch.bool
    assert batch.age_bin is None  # no bin edges configured

    for i in range(len(pyd)):
        expected = _expected_sm_window_ages(pyd, i, raw_event_ages)
        got = batch.age_years[i][batch.age_mask[i]].numpy()
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3)
        # Padding positions are masked and zero-filled.
        assert batch.age_mask[i].sum() == len(expected)
        assert (batch.age_years[i][~batch.age_mask[i]] == 0).all()
        valid_positions = torch.nonzero(batch.age_mask[i]).squeeze(1)
        if padding_side == "right":
            assert valid_positions.tolist() == list(range(len(expected)))
        else:
            width = batch.age_mask.shape[1]
            assert valid_positions.tolist() == list(range(width - len(expected), width))


def test_sem_per_position_age_is_event_level_and_masks_static_event(sample_dataset_config, raw_event_ages):
    cfg = _age_cfg(sample_dataset_config, include_age=True, batch_mode="SEM", static_inclusion_mode="prepend")
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)

    assert batch.age_years.shape == batch.event_mask.shape
    # The prepended static event never carries an age.
    assert not batch.age_mask[batch.static_mask].any()
    assert torch.equal(batch.age_mask, batch.event_mask & ~batch.static_mask)

    for i, (subject_id, end_idx) in enumerate(pyd.index):
        got = batch.age_years[i][batch.age_mask[i]].numpy()
        expected = raw_event_ages[subject_id][:end_idx][-len(got) :]
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3)


def test_age_bins(sample_dataset_config):
    cfg = _age_cfg(sample_dataset_config, include_age=True, age_bin_edges_years=[1, 30, 33])
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)

    valid_ages = batch.age_years[batch.age_mask].numpy()
    np.testing.assert_array_equal(
        batch.age_bin[batch.age_mask].numpy(), np.digitize(valid_ages, [1, 30, 33]) + 1
    )
    assert (batch.age_bin[~batch.age_mask] == 0).all()
    assert batch.age_bin.max() <= cfg.n_age_bins


def test_numeric_age_token_in_include_mode(sample_dataset_config, raw_event_ages):
    cfg = _age_cfg(sample_dataset_config, age_token_mode="numeric")
    pyd = MEDSPytorchDataset(cfg, split="train")
    plain = MEDSPytorchDataset(_age_cfg(sample_dataset_config), split="train")
    batch = _full_batch(pyd)
    plain_batch = _full_batch(plain)

    # The age token is appended after the real static measurements...
    n_static = plain_batch.static_code.shape[1]
    assert batch.static_code.shape[1] == n_static + 1
    torch.testing.assert_close(batch.static_code[:, :n_static], plain_batch.static_code)
    assert (batch.static_code[:, n_static] == cfg.base_vocab_size).all()
    assert cfg.derived_codes == {cfg.base_vocab_size: "AGE"}
    assert cfg.vocab_size == cfg.base_vocab_size + 1

    # ...the dynamic data is unchanged, and no per-position ages are emitted.
    torch.testing.assert_close(batch.code, plain_batch.code)
    assert batch.age_years is None

    # ...and carries the age at the first event of the sampled window.
    for i in range(len(pyd)):
        expected_start_age = _expected_sm_window_ages(pyd, i, raw_event_ages)[0]
        assert batch.static_numeric_value_mask[i, n_static]
        assert batch.static_numeric_value[i, n_static].item() == pytest.approx(expected_start_age, abs=1e-3)


@pytest.mark.parametrize("batch_mode", ["SM", "SEM"])
def test_quantized_age_token_in_prepend_mode(sample_dataset_config, raw_event_ages, batch_mode):
    edges = [20, 30, 33]
    cfg = _age_cfg(
        sample_dataset_config,
        age_token_mode="quantized",
        age_bin_edges_years=edges,
        static_inclusion_mode="prepend",
        batch_mode=batch_mode,
        max_seq_len=6 if batch_mode == "SM" else 3,
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)
    first_derived = cfg.base_vocab_size

    # The sequence budget still holds with the extra static element.
    assert batch.code.shape[1] <= cfg.max_seq_len

    for i, (subject_id, end_idx) in enumerate(pyd.index):
        if batch_mode == "SM":
            static_positions = torch.nonzero(batch.static_mask[i]).squeeze(1)
            age_code = batch.code[i, static_positions[-1]].item()
            start_age = _expected_sm_window_ages(pyd, i, raw_event_ages)[0]
        else:
            static_event = batch.code[i, 0][batch.code[i, 0] != 0]
            age_code = static_event[-1].item()
            n_dynamic_events = int((batch.event_mask[i] & ~batch.static_mask[i]).sum())
            start_age = raw_event_ages[subject_id][:end_idx][-n_dynamic_events]

        expected_bin = int(np.digitize([start_age], edges)[0])
        assert age_code == first_derived + expected_bin
        assert cfg.derived_codes[age_code].startswith("AGE//")


def test_step_through_sm_prepend_reserves_age_token_slot(sample_dataset_config):
    cfg = _age_cfg(
        sample_dataset_config,
        age_token_mode="numeric",
        static_inclusion_mode="prepend",
        seq_sampling_strategy="step_through",
        step_through_overlap=0,
        max_seq_len=5,
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    for i in range(len(pyd)):
        item = pyd[i]
        assert len(item["dynamic"]) <= cfg.max_seq_len
        assert item["n_static_seq_els"] == 3  # 2 real static measurements + the age token


def test_unknown_birth_codes_fail_at_construction(sample_dataset_config):
    cfg = _age_cfg(sample_dataset_config, include_age=True, birth_codes=["MEDS_BIRTH"])
    with pytest.raises(ValueError, match="None of birth_codes"):
        MEDSPytorchDataset(cfg, split="train")


def test_age_features_through_dataloader_workers(sample_dataset_config):
    cfg = _age_cfg(
        sample_dataset_config, include_age=True, age_token_mode="numeric", age_bin_edges_years=[30]
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    expected = _full_batch(pyd)
    (batch,) = list(pyd.get_dataloader(batch_size=len(pyd), shuffle=False, num_workers=2))
    for name in ("age_years", "age_mask", "age_bin", "static_code", "static_numeric_value"):
        torch.testing.assert_close(batch[name], expected[name])


def test_unknown_age_is_masked_and_gets_unknown_token(sample_dataset_config):
    # Use a code only some subjects have as the "birth" code, so the others have unknown ages.
    cfg = _age_cfg(
        sample_dataset_config,
        birth_codes=["ADMISSION//CARDIAC"],
        include_age=True,
        age_token_mode="quantized",
        age_bin_edges_years=[1],
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)
    unknown_code = cfg.base_vocab_size + cfg.n_age_bins
    assert cfg.derived_codes[unknown_code] == "AGE//UNKNOWN"
    (birth_idx,) = cfg.birth_vocab_indices

    n_known = 0
    for i, (subject_id, end_idx) in enumerate(pyd.index):
        dynamic, _ = pyd.load_subject_data(subject_id, 0, end_idx)
        has_birth = birth_idx in dynamic.tensors["dim1/code"]
        n_known += has_birth
        assert bool(batch.age_mask[i].any()) == has_birth
        assert (batch.static_code[i, -1].item() == unknown_code) == (not has_birth)
    assert 0 < n_known < len(pyd), "fixture should mix known- and unknown-age subjects"
