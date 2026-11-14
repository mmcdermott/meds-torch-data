"""Tests for the time-derived age measurements (`include_age`, `age_token_mode`, `n_age_bins`).

Expected values are computed independently from the raw MEDS data (the `DOB` row and the unique event
times per subject) — including the training-set age statistics used for standardization and binning —
rather than from anything the preprocessing pipeline or the dataset itself produced.
"""

import dataclasses
import shutil

import numpy as np
import polars as pl
import pytest
import torch
from meds import train_split
from meds_testing_helpers.dataset import MEDSDataset

from meds_torchdata import MEDSPytorchDataset

DAYS_PER_YEAR = 365.25


@pytest.fixture(scope="module")
def raw_events(simple_static_MEDS) -> pl.DataFrame:
    """One row per (subject, unique dynamic event time), with the subject's split and age in years."""
    D = MEDSDataset(root_dir=simple_static_MEDS)
    df = pl.concat(
        [
            pl.from_arrow(shard).with_columns(pl.lit(name.split("/")[0]).alias("split"))
            for name, shard in D.data_shards.items()
        ]
    )
    dob = df.filter(pl.col("code") == "DOB").select("subject_id", pl.col("time").alias("dob"))
    return (
        df.filter(pl.col("time").is_not_null())
        .select("subject_id", "split", "time")
        .unique()
        .join(dob, on="subject_id")
        .with_columns(
            ((pl.col("time") - pl.col("dob")).dt.total_seconds() / 86400 / DAYS_PER_YEAR).alias("age")
        )
        .sort("subject_id", "time")
    )


@pytest.fixture(scope="module")
def raw_event_ages(raw_events) -> dict[int, np.ndarray]:
    return {
        sid: grp["age"].to_numpy() for (sid,), grp in raw_events.group_by("subject_id", maintain_order=True)
    }


@pytest.fixture(scope="module")
def raw_train_ages(raw_events) -> np.ndarray:
    """Training-set ages over non-birth events (the population the preprocessing stats describe)."""
    train = raw_events.filter((pl.col("split") == train_split) & (pl.col("time") != pl.col("dob")))
    return train["age"].to_numpy()


@pytest.fixture(scope="module")
def raw_age_stats(raw_train_ages) -> tuple[float, float]:
    return float(raw_train_ages.mean()), float(raw_train_ages.std())


def _age_cfg(sample_dataset_config, **kwargs):
    defaults = {"seq_sampling_strategy": "to_end", "max_seq_len": 6}
    return dataclasses.replace(sample_dataset_config, **{**defaults, **kwargs})


def _full_batch(pyd: MEDSPytorchDataset):
    return pyd.collate([pyd[i] for i in range(len(pyd))])


def _dynamic_mask(batch) -> torch.Tensor:
    """Positions holding dynamic data: real (non-padding) positions neither static nor time-derived."""
    real = batch.event_mask if batch.event_mask is not None else batch.code != 0
    for mask in (batch.static_mask, batch.time_derived_mask):
        if mask is not None:
            real = real & ~mask
    return real


def _expected_sm_window_ages(pyd: MEDSPytorchDataset, idx: int, raw_ages: dict) -> np.ndarray:
    """Raw age at each measurement of the sampled (TO_END) window for dataset element `idx` in SM mode."""
    subject_id, end_idx = pyd.index[idx]
    dynamic, _ = pyd.load_subject_data(subject_id, 0, end_idx)
    meas_per_event = np.diff(dynamic.tensors["dim1/bounds"], prepend=0)
    per_meas = np.repeat(raw_ages[subject_id][:end_idx], meas_per_event)
    item = pyd[idx]
    n_window = len(item["dynamic"]) - item.get("n_static_seq_els", 0)
    return per_meas[len(per_meas) - n_window :]


def test_preprocessed_age_stats_match_raw_data(sample_dataset_config, raw_age_stats):
    mean, std = raw_age_stats
    stats = sample_dataset_config.age_stats
    assert stats["mean"] == pytest.approx(mean, abs=1e-4)
    assert stats["std"] == pytest.approx(std, abs=1e-4)


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_sm_per_position_age_is_standardized_and_matches_raw_data(
    sample_dataset_config, raw_event_ages, raw_age_stats, padding_side
):
    mean, std = raw_age_stats
    cfg = _age_cfg(sample_dataset_config, include_age=True, padding_side=padding_side)
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)

    assert batch.age.shape == batch.code.shape
    assert batch.age_bin is None  # no bins configured
    dynamic = _dynamic_mask(batch)
    assert (batch.age[~dynamic] == 0).all()

    for i in range(len(pyd)):
        expected = (_expected_sm_window_ages(pyd, i, raw_event_ages) - mean) / std
        np.testing.assert_allclose(batch.age[i][dynamic[i]].numpy(), expected, rtol=0, atol=1e-3)


def test_sem_per_position_age_is_event_level(sample_dataset_config, raw_event_ages, raw_age_stats):
    mean, std = raw_age_stats
    cfg = _age_cfg(sample_dataset_config, include_age=True, batch_mode="SEM", static_inclusion_mode="prepend")
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)

    assert batch.age.shape == batch.event_mask.shape
    dynamic = _dynamic_mask(batch)
    assert (batch.age[~dynamic] == 0).all()
    for i, (subject_id, end_idx) in enumerate(pyd.index):
        got = batch.age[i][dynamic[i]].numpy()
        expected = (raw_event_ages[subject_id][:end_idx][-len(got) :] - mean) / std
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3)


def test_per_position_age_bins_are_equal_frequency_quantiles(sample_dataset_config, raw_train_ages):
    n_bins = 4
    cfg = _age_cfg(sample_dataset_config, include_age=True, n_age_bins=n_bins)
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)

    # The edges sit at the training ages' n-tiles (to within the 1-day histogram resolution).
    expected_edges = np.quantile(raw_train_ages, np.arange(1, n_bins) / n_bins, method="inverted_cdf")
    np.testing.assert_allclose(cfg.age_bin_edges, expected_edges, rtol=0, atol=1 / DAYS_PER_YEAR)

    # Per-position bins are 1-indexed at dynamic positions, 0 elsewhere.
    dynamic = _dynamic_mask(batch)
    assert (batch.age_bin[~dynamic] == 0).all()
    assert ((batch.age_bin[dynamic] >= 1) & (batch.age_bin[dynamic] <= n_bins)).all()
    raw_ages = batch.age[dynamic].numpy() * cfg.age_stats["std"] + cfg.age_stats["mean"]
    np.testing.assert_array_equal(batch.age_bin[dynamic].numpy(), cfg.quantize_age(raw_ages) + 1)


@pytest.mark.parametrize("static_inclusion_mode", ["omit", "include", "prepend"])
@pytest.mark.parametrize("batch_mode", ["SM", "SEM"])
def test_numeric_age_token_is_prepended_in_every_static_mode(
    sample_dataset_config, raw_event_ages, raw_age_stats, static_inclusion_mode, batch_mode
):
    mean, std = raw_age_stats
    kwargs = {"static_inclusion_mode": static_inclusion_mode, "batch_mode": batch_mode}
    if batch_mode == "SEM":
        kwargs["max_seq_len"] = 4
    cfg = _age_cfg(sample_dataset_config, age_token_mode="numeric", **kwargs)
    pyd = MEDSPytorchDataset(cfg, split="train")
    plain = MEDSPytorchDataset(_age_cfg(sample_dataset_config, **kwargs), split="train")
    batch = _full_batch(pyd)
    plain_batch = _full_batch(plain)

    # Exactly one time-derived position per sample, disjoint from the static data, and never part of the
    # static_code tensors in INCLUDE mode.
    td = batch.time_derived_mask
    assert (td.sum(dim=1) == 1).all()
    if batch.static_mask is not None:
        assert not (td & batch.static_mask).any()
    if static_inclusion_mode == "include":
        torch.testing.assert_close(batch.static_code, plain_batch.static_code)
    assert cfg.time_derived_codes == {cfg.base_vocab_size: "AGE"}
    assert cfg.vocab_size == cfg.base_vocab_size + 1

    # The sequence budget is respected.
    assert batch.code.shape[1] <= cfg.max_seq_len

    for i, (subject_id, end_idx) in enumerate(pyd.index):
        pos = int(torch.nonzero(td[i]).item())
        if batch_mode == "SM":
            code, value = batch.code[i, pos].item(), batch.numeric_value[i, pos].item()
            start_age = _expected_sm_window_ages(pyd, i, raw_event_ages)[0]
        else:
            event_codes = batch.code[i, pos][batch.code[i, pos] != 0]
            assert len(event_codes) == 1
            code = event_codes.item()
            value = batch.numeric_value[i, pos, 0].item()
            n_dynamic = int(_dynamic_mask(batch)[i].sum())
            start_age = raw_event_ages[subject_id][:end_idx][-n_dynamic]
        assert code == cfg.base_vocab_size
        assert value == pytest.approx((start_age - mean) / std, abs=1e-3)


def test_quantized_age_token(sample_dataset_config, raw_event_ages):
    cfg = _age_cfg(
        sample_dataset_config, age_token_mode="quantized", n_age_bins=3, static_inclusion_mode="prepend"
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    batch = _full_batch(pyd)
    assert len(cfg.time_derived_codes) == 4  # 3 bins + unknown
    for i in range(len(pyd)):
        pos = int(torch.nonzero(batch.time_derived_mask[i]).item())
        start_age = _expected_sm_window_ages(pyd, i, raw_event_ages)[0]
        assert batch.code[i, pos].item() == cfg.base_vocab_size + int(cfg.quantize_age([start_age])[0])
        assert not batch.numeric_value_mask[i, pos]


@pytest.mark.parametrize("batch_mode", ["SM", "SEM"])
def test_static_and_time_derived_masks_respect_left_padding(sample_dataset_config, batch_mode):
    """Regression test: `static_mask` used to ignore `padding_side` and always mark the first positions."""
    cfg = _age_cfg(
        sample_dataset_config,
        batch_mode=batch_mode,
        static_inclusion_mode="prepend",
        age_token_mode="numeric",
        padding_side="left",
        max_seq_len=10,
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    items = [pyd[i] for i in range(len(pyd))]
    batch = pyd.collate(items)
    seq_len = batch.code.shape[1]
    assert any(len(item["dynamic"]) + 1 < seq_len for item in items), "fixture should include padded samples"
    for i, item in enumerate(items):
        pad = seq_len - len(item["dynamic"]) - 1  # the time-derived block is inserted at collation
        n_static = item["n_static_seq_els"]
        assert batch.static_mask[i].nonzero().squeeze(1).tolist() == list(range(pad, pad + n_static))
        assert batch.time_derived_mask[i].nonzero().squeeze(1).tolist() == [pad + n_static]


def test_step_through_reserves_time_derived_slot(sample_dataset_config):
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
        assert item["n_static_seq_els"] == 2
        assert len(item["time_derived_code"]) == 1
        assert len(item["dynamic"]) + 1 <= cfg.max_seq_len  # + the time-derived block


def test_unknown_birth_is_neutral(sample_dataset_config):
    """Subjects without a birth event get age 0 / bin 0 everywhere and the `AGE//UNKNOWN` token."""
    cfg = _age_cfg(
        sample_dataset_config, include_age=True, age_token_mode="quantized", n_age_bins=3, max_seq_len=10
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    # Simulate a subject without a birth event by blanking one subject's precomputed ages.
    subject_id = pyd.index[0][0]
    shard, subject_idx = pyd.subj_locations[subject_id]
    ages, starts = pyd._event_ages_by_shard[shard]
    n_events = len(pyd.schema_dfs_by_shard[shard][subject_idx]["time"].item())
    ages[starts[subject_idx] : starts[subject_idx] + n_events] = np.nan

    batch = _full_batch(pyd)
    unknown_code = cfg.base_vocab_size + cfg.n_age_bins
    assert cfg.time_derived_codes[unknown_code] == "AGE//UNKNOWN"
    assert (batch.age[0] == 0).all() and (batch.age_bin[0] == 0).all()
    assert batch.code[0][batch.time_derived_mask[0]].item() == unknown_code
    assert (batch.age_bin[1:][_dynamic_mask(batch)[1:]] > 0).all()


def test_age_features_require_preprocessed_birth_times(sample_dataset_config, tmp_path):
    cohort = tmp_path / "cohort"
    shutil.copytree(sample_dataset_config.tensorized_cohort_dir, cohort)
    for fp in (cohort / "tokenization" / "schemas").rglob("*.parquet"):
        pl.read_parquet(fp).drop("birth_time").write_parquet(fp)
    cfg = dataclasses.replace(sample_dataset_config, tensorized_cohort_dir=cohort, include_age=True)
    with pytest.raises(ValueError, match="Re-run preprocessing"):
        MEDSPytorchDataset(cfg, split="train")


def test_age_features_through_dataloader_workers(sample_dataset_config):
    cfg = _age_cfg(sample_dataset_config, include_age=True, age_token_mode="numeric", n_age_bins=3)
    pyd = MEDSPytorchDataset(cfg, split="train")
    expected = _full_batch(pyd)
    (batch,) = list(pyd.get_dataloader(batch_size=len(pyd), shuffle=False, num_workers=2))
    for name in ("code", "numeric_value", "age", "age_bin", "time_derived_mask"):
        torch.testing.assert_close(batch[name], expected[name])
