"""Tests for the STEP_THROUGH subsequence sampling strategy."""

import logging

import pytest
import torch

from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
from meds_torchdata.types import SubsequenceSamplingStrategy


def test_step_through_requires_positive_stride(tensorized_MEDS_dataset):
    with pytest.raises(
        ValueError, match="Exactly one of step_through_stride or step_through_overlap must be set"
    ):
        MEDSTorchDataConfig(
            tensorized_cohort_dir=tensorized_MEDS_dataset,
            max_seq_len=3,
            seq_sampling_strategy="step_through",
        )
    with pytest.raises(ValueError, match="step_through_stride must be a positive integer"):
        MEDSTorchDataConfig(
            tensorized_cohort_dir=tensorized_MEDS_dataset,
            max_seq_len=3,
            seq_sampling_strategy="step_through",
            step_through_stride=0,
        )


def test_step_through_overlap_accepted(tensorized_MEDS_dataset):
    # `step_through_overlap=0` = contiguous windows; equivalent to stride == effective_window.
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_overlap=0,
        batch_mode="SEM",
    )
    dataset = MEDSPytorchDataset(cfg, split="train")
    assert len(dataset) > 0


def test_step_through_overlap_rejects_bool_and_negative(tensorized_MEDS_dataset):
    for bad in (True, False, -1):
        with pytest.raises(ValueError, match="step_through_overlap must be a non-negative integer"):
            MEDSTorchDataConfig(
                tensorized_cohort_dir=tensorized_MEDS_dataset,
                max_seq_len=3,
                seq_sampling_strategy="step_through",
                step_through_overlap=bad,
            )


def test_step_through_stride_and_overlap_mutually_exclusive(tensorized_MEDS_dataset):
    with pytest.raises(ValueError, match="Exactly one of step_through_stride or step_through_overlap"):
        MEDSTorchDataConfig(
            tensorized_cohort_dir=tensorized_MEDS_dataset,
            max_seq_len=3,
            seq_sampling_strategy="step_through",
            step_through_stride=2,
            step_through_overlap=1,
        )


def test_step_through_stride_rejects_bool(tensorized_MEDS_dataset):
    # `bool` is a subclass of `int` in Python, so a naive `isinstance(x, int)` check would
    # silently accept `True` / `False` as stride values. Config validation rejects them.
    for bad_value in (True, False):
        with pytest.raises(ValueError, match="step_through_stride must be a positive integer"):
            MEDSTorchDataConfig(
                tensorized_cohort_dir=tensorized_MEDS_dataset,
                max_seq_len=3,
                seq_sampling_strategy="step_through",
                step_through_stride=bad_value,
            )


def test_step_through_stride_requires_step_through_strategy(tensorized_MEDS_dataset):
    with pytest.raises(ValueError, match="may only be set when seq_sampling_strategy is STEP_THROUGH"):
        MEDSTorchDataConfig(
            tensorized_cohort_dir=tensorized_MEDS_dataset,
            max_seq_len=3,
            seq_sampling_strategy="random",
            step_through_stride=2,
        )


def test_step_through_subsample_st_offset_delegates_to_to_end():
    # STEP_THROUGH delegates to TO_END semantics at the sampler level: the dataset has
    # already modified the index to point to the right prefix endpoint, and the sampler
    # takes the last `max_seq_len` elements of it.
    assert SubsequenceSamplingStrategy.STEP_THROUGH.subsample_st_offset(10, 3) == 7
    assert SubsequenceSamplingStrategy.STEP_THROUGH.subsample_st_offset(10, 3) == (
        SubsequenceSamplingStrategy.TO_END.subsample_st_offset(10, 3)
    )
    assert SubsequenceSamplingStrategy.STEP_THROUGH.subsample_st_offset(5, 10) is None


def _step_through_cfg(tensorized_MEDS_dataset, **kwargs):
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_stride=2,
        batch_mode="SEM",
        **kwargs,
    )


def test_step_through_expands_index_and_emits_warning(tensorized_MEDS_dataset, caplog):
    cfg = _step_through_cfg(tensorized_MEDS_dataset)

    with caplog.at_level(logging.WARNING, logger="meds_torchdata.pytorch_dataset"):
        dataset = MEDSPytorchDataset(cfg, split="train")

    # The warning fires *after* the expansion loop so the numbers it reports are the actual
    # observed per-subject window counts (in PREPEND mode with per-subject effective windows
    # a closed-form formula in terms of `config.max_seq_len` would be misleading).
    warning_messages = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("STEP_THROUGH sampling expanded" in msg for msg in warning_messages), (
        f"Expected oversampling warning on dataset __init__, got: {warning_messages}"
    )
    assert any("n_subject_windows" in msg for msg in warning_messages), (
        "Expected the warning to mention the n_subject_windows reweighting escape hatch"
    )

    # Expansion must preserve the full set of subjects and produce at least as many elements
    # as there are subjects; subjects with long sequences should contribute more than one.
    n_subjects = len({s for s, _ in dataset.index})
    assert len(dataset) >= n_subjects
    assert dataset._windows_per_subject is not None
    assert set(dataset._windows_per_subject) == {s for s, _ in dataset.index}
    # Parallel meas-ends list is `None` in SEM mode (events are atomic there, and the
    # index's event-level `end` plus TO_END sampling suffices).
    assert dataset.step_through_meas_ends is None

    # At least one subject must have been expanded into more than one window for the test
    # to be meaningful.
    assert max(dataset._windows_per_subject.values()) > 1

    # Consecutive entries belonging to the same subject must have strictly non-decreasing
    # window ends (deterministic ordered walk).
    last_subject = None
    last_end = -1
    for subject_id, end in dataset.index:
        if subject_id != last_subject:
            last_subject = subject_id
            last_end = end
            continue
        assert end >= last_end, f"Window ends must be monotonically non-decreasing for subject {subject_id}"
        last_end = end


def test_step_through_walks_all_events_at_least_once(tensorized_MEDS_dataset):
    # `step_through_overlap=0` -> contiguous non-overlapping walk in events. Every event
    # from 0 to the subject's end must appear in at least one window's effective range.
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_overlap=0,
        batch_mode="SEM",
    )
    dataset = MEDSPytorchDataset(cfg, split="train")

    # Group index entries by subject. In SEM mode, each index entry's `end` is the event
    # end of the window, and the window contains events [max(0, end - effective_window), end).
    per_subject: dict[int, list[int]] = {}
    for subject_id, end in dataset.index:
        per_subject.setdefault(subject_id, []).append(end)

    for subject_id, ends in per_subject.items():
        effective_window = dataset._effective_max_seq_len_for(subject_id)
        covered: set[int] = set()
        for end in ends:
            covered.update(range(max(0, end - effective_window), end))
        # The subject's full event range must be covered (including the tail-anchored window).
        max_end = max(ends)
        for pos in range(max_end):
            assert pos in covered, f"Event {pos} for subject {subject_id} not covered"


def test_step_through_window_counts_match_loss_reweighting_contract(tensorized_MEDS_dataset):
    cfg = _step_through_cfg(tensorized_MEDS_dataset, include_subject_window_counts_in_batch=True)
    dataset = MEDSPytorchDataset(cfg, split="train")

    # Collate a batch containing every element so we cover multiple subjects.
    samples = [dataset[i] for i in range(len(dataset))]
    batch = dataset.collate(samples)

    assert batch.n_subject_windows is not None
    assert batch.n_subject_windows.dtype == torch.long
    assert batch.n_subject_windows.shape == (len(dataset),)

    # The per-element count equals the number of elements in the dataset that share the same
    # subject (that is the definition of the reweighting denominator).
    from collections import Counter

    subj_counts = Counter(subject_id for subject_id, _ in dataset.index)
    for i, (subject_id, _) in enumerate(dataset.index):
        assert int(batch.n_subject_windows[i]) == subj_counts[subject_id]

    # Sum of 1 / n_subject_windows across the dataset equals the number of unique subjects:
    # this is the property that makes reweighted losses unbiased w.r.t. oversampling.
    reweight_sum = float((1.0 / batch.n_subject_windows.float()).sum().item())
    assert reweight_sum == pytest.approx(len(set(subj_counts)), abs=1e-6)


def test_include_subject_window_counts_without_step_through(tensorized_MEDS_dataset):
    # Non step-through datasets can still opt in; each sample reports a count of 1.
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="from_start",
        include_subject_window_counts_in_batch=True,
    )
    dataset = MEDSPytorchDataset(cfg, split="train")
    batch = dataset.collate([dataset[i] for i in range(len(dataset))])

    assert batch.n_subject_windows is not None
    assert torch.equal(batch.n_subject_windows, torch.ones(len(dataset), dtype=torch.long))


@pytest.mark.parametrize("batch_mode", ["SEM", "SM"])
def test_step_through_prepend_respects_max_seq_len(tensorized_MEDS_dataset, batch_mode):
    """STEP_THROUGH + PREPEND must not produce samples longer than `config.max_seq_len`.

    In PREPEND mode the collated sample is `[static; dynamic]`, so the dynamic window size
    has to be reduced by the number of static elements being prepended. Prior to the fix the
    step-through expansion precomputed windows of size `max_seq_len` regardless of mode,
    which meant the concatenated `[static; dynamic]` sample could exceed `max_seq_len` when
    `static_inclusion_mode=PREPEND`.
    """

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=8,
        seq_sampling_strategy="step_through",
        step_through_stride=2,
        batch_mode=batch_mode,
        static_inclusion_mode="prepend",
    )
    dataset = MEDSPytorchDataset(cfg, split="train")

    # Every single sample — after prepending static data — must have length <= max_seq_len.
    for idx in range(len(dataset)):
        sample = dataset[idx]
        total_len = len(sample["dynamic"])
        assert total_len <= cfg.max_seq_len, (
            f"Sample {idx} total length {total_len} exceeds max_seq_len={cfg.max_seq_len} "
            f"under batch_mode={batch_mode}, static_inclusion_mode=prepend"
        )

    # And the collated batch must also fit.
    batch = dataset.collate([dataset[i] for i in range(len(dataset))])
    seq_axis = batch.time_delta_days.shape[-1]
    assert seq_axis <= cfg.max_seq_len, (
        f"Collated batch shape {batch.time_delta_days.shape} exceeds max_seq_len={cfg.max_seq_len}"
    )


def test_step_through_prepend_effective_window_matches_reduction(tensorized_MEDS_dataset):
    """The per-subject effective window is exactly `max_seq_len - n_static_seq_els`."""

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=8,
        seq_sampling_strategy="step_through",
        step_through_stride=2,
        batch_mode="SM",
        static_inclusion_mode="prepend",
    )
    dataset = MEDSPytorchDataset(cfg, split="train")

    for subject_id, _ in set(dataset.index):
        effective = dataset._effective_max_seq_len_for(subject_id)
        shard, subj_idx = dataset.subj_locations[subject_id]
        static_code_list = dataset.schema_dfs_by_shard[shard][subj_idx]["static_code"].item()
        n_static = len(static_code_list) if static_code_list is not None else 0
        assert effective == cfg.max_seq_len - n_static


def test_step_through_sm_mode_uses_measurement_units(tensorized_MEDS_dataset):
    # In SM mode, the window must be computed in post-flatten measurement units, not in
    # events. We verify that explicitly by checking at least one returned window is as long
    # as `max_seq_len` measurements.
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_stride=2,
        batch_mode="SM",
    )
    dataset = MEDSPytorchDataset(cfg, split="train")

    # Iterate through every sample and check the dynamic data length is <= max_seq_len.
    for i in range(len(dataset)):
        sample = dataset[i]
        dyn = sample["dynamic"]
        # In SM mode process_dynamic_data flattens before slicing; the returned length must
        # never exceed the configured max_seq_len.
        assert len(dyn) <= cfg.max_seq_len, (
            f"Sample {i} returned a window of length {len(dyn)} which exceeds max_seq_len={cfg.max_seq_len}"
        )

    # And at least one window should realize the full max_seq_len (assuming the test fixture
    # has a subject with more than `max_seq_len` measurements, which it does).
    any_full = any(len(dataset[i]["dynamic"]) == cfg.max_seq_len for i in range(len(dataset)))
    assert any_full
