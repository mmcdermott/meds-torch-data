"""Tests for the STEP_THROUGH subsequence sampling strategy.

Most of the fine-grained behavior for `STEP_THROUGH` is covered by doctests:

- `MEDSTorchDataConfig` doctest: every config validation error case
  (stride/overlap types, mutual exclusivity, wrong-strategy rejection, bool rejection).
- `SubsequenceSamplingStrategy.subsample_st_offset` doctest: the `STEP_THROUGH → TO_END`
  delegation.
- `MEDSPytorchDataset._expand_index_for_step_through` doctest: a SEM-mode end-to-end walk
  with `self.index` / `_windows_per_subject` / per-sample dynamic payload / collated
  `n_subject_windows` tensor, plus an SM-mode walk showing the Design B property that
  windows can end mid-event.
- `MEDSPytorchDataset._effective_max_seq_len_for` doctest: SEM / SM / PREPEND per-subject
  reductions.
- `MEDSPytorchDataset._step_through_event_ends_sem` doctest: the SEM event-walk closed
  form.
- `MEDSPytorchDataset._step_through_ends_sm` doctest: the SM `searchsorted` walk on the
  real fixture.

This file keeps only the assertions that the doctest format cannot reasonably express —
things that need `caplog` inspection, `pytest.parametrize`, or iterating across the whole
dataset with algorithmic coverage checks.
"""

import logging
from collections import Counter

import polars as pl
import pytest
import torch

from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig


def _step_through_cfg(tensorized_MEDS_dataset, **kwargs):
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_stride=2,
        batch_mode="SEM",
        **kwargs,
    )


def test_step_through_emits_oversampling_warning(tensorized_MEDS_dataset, caplog):
    """The dataset logs a `STEP_THROUGH sampling expanded ...` warning on `__init__`.

    This has to be a real test rather than a doctest because it inspects `caplog.records`.
    """
    cfg = _step_through_cfg(tensorized_MEDS_dataset)

    with caplog.at_level(logging.WARNING, logger="meds_torchdata.pytorch_dataset"):
        MEDSPytorchDataset(cfg, split="train")

    warning_messages = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("STEP_THROUGH sampling expanded" in msg for msg in warning_messages), (
        f"Expected oversampling warning on dataset __init__, got: {warning_messages}"
    )
    assert any("n_subject_windows" in msg for msg in warning_messages), (
        "Expected the warning to mention the n_subject_windows reweighting escape hatch"
    )


def test_step_through_overlap_zero_covers_every_event(tensorized_MEDS_dataset):
    """With `step_through_overlap=0` every event in every subject must appear in some window.

    The closed form for the event walk is covered by `_step_through_event_ends_sem`'s
    doctest, but verifying "no event is left behind across the *whole* dataset" requires
    iterating every subject, which doesn't fit cleanly into a doctest.
    """
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_overlap=0,
        batch_mode="SEM",
    )
    dataset = MEDSPytorchDataset(cfg, split="train")

    per_subject: dict[int, list[int]] = {}
    for subject_id, end in dataset.index:
        per_subject.setdefault(subject_id, []).append(end)

    for subject_id, ends in per_subject.items():
        effective_window = dataset._effective_max_seq_len_for(subject_id)
        covered: set[int] = set()
        for end in ends:
            covered.update(range(max(0, end - effective_window), end))
        max_end = max(ends)
        for pos in range(max_end):
            assert pos in covered, f"Event {pos} for subject {subject_id} not covered"


def test_step_through_sm_errors_on_old_cohort_missing_column(tensorized_MEDS_dataset, tmp_path):
    """Older tensorized cohorts without `measurements_per_event` get a clear migration error.

    Build a mock "old cohort" by copying the current tensorized cohort and stripping the
    `measurements_per_event` column from every schema parquet, then point an SM-mode
    step-through config at it. The dataset must raise a `ValueError` mentioning
    `MTD_preprocess`, not a low-level parquet/column-not-found traceback. This guards the
    eager-read migration path in `MEDSPytorchDataset.__init__`.
    """
    import shutil

    old_cohort = tmp_path / "old_cohort"
    shutil.copytree(tensorized_MEDS_dataset, old_cohort)

    # Rewrite every schema parquet without the `measurements_per_event` column.
    stripped_any = False
    for schema_fp in (old_cohort / "tokenization" / "schemas").rglob("*.parquet"):
        df = pl.read_parquet(schema_fp)
        if "measurements_per_event" in df.columns:
            df.drop("measurements_per_event").write_parquet(schema_fp)
            stripped_any = True
    assert stripped_any, "Test setup expected the current cohort to have the new column"

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=old_cohort,
        max_seq_len=5,
        seq_sampling_strategy="step_through",
        step_through_stride=3,
        batch_mode="SM",
    )
    with pytest.raises(ValueError) as excinfo:
        MEDSPytorchDataset(cfg, split="train")
    msg = str(excinfo.value)
    assert "STEP_THROUGH sampling in SM mode requires" in msg
    assert "measurements_per_event" in msg
    assert "MTD_preprocess" in msg

    # SEM mode and non-step-through configs do *not* need the new column, so they keep
    # working on the stripped cohort — this is the backward-compat property the eager
    # column check is there to preserve.
    sem_cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=old_cohort,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_stride=2,
        batch_mode="SEM",
    )
    MEDSPytorchDataset(sem_cfg, split="train")  # no error

    random_sm_cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=old_cohort,
        max_seq_len=5,
        seq_sampling_strategy="random",
        batch_mode="SM",
    )
    MEDSPytorchDataset(random_sm_cfg, split="train")  # no error


def test_step_through_overlap_too_large_rejected(tensorized_MEDS_dataset):
    """`step_through_overlap >= effective_window` is rejected at dataset __init__."""
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_overlap=3,  # == max_seq_len, so effective_window - overlap == 0
        batch_mode="SEM",
    )
    with pytest.raises(ValueError, match=r"step_through_overlap .* must be strictly less than"):
        MEDSPytorchDataset(cfg, split="train")


def test_step_through_window_counts_match_loss_reweighting_contract(tensorized_MEDS_dataset):
    """`batch.n_subject_windows` satisfies the `sum(1/n) == #subjects` reweighting invariant.

    The sample-level `n_subject_windows` lookup is shown in the
    `_expand_index_for_step_through` doctest, but the reweighting invariant
    (`sum(1 / n_subject_windows) == #unique_subjects`) is a property of the whole collated
    batch — it's the definition of what makes the reweighting unbiased w.r.t. oversampling.
    """
    cfg = _step_through_cfg(tensorized_MEDS_dataset, include_subject_window_counts_in_batch=True)
    dataset = MEDSPytorchDataset(cfg, split="train")

    samples = [dataset[i] for i in range(len(dataset))]
    batch = dataset.collate(samples)

    assert batch.n_subject_windows is not None
    assert batch.n_subject_windows.dtype == torch.long
    assert batch.n_subject_windows.shape == (len(dataset),)

    subj_counts = Counter(subject_id for subject_id, _ in dataset.index)
    for i, (subject_id, _) in enumerate(dataset.index):
        assert int(batch.n_subject_windows[i]) == subj_counts[subject_id]

    reweight_sum = float((1.0 / batch.n_subject_windows.float()).sum().item())
    assert reweight_sum == pytest.approx(len(set(subj_counts)), abs=1e-6)


def test_include_subject_window_counts_without_step_through(tensorized_MEDS_dataset):
    """Non-step-through datasets that opt into the flag report a uniform count of 1."""
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

    Parametrized across `batch_mode` because SEM reserves exactly one event for the static
    data while SM reserves `len(static_code)` measurements per subject, and both branches
    need to round-trip end-to-end through `collate`.
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

    for idx in range(len(dataset)):
        sample = dataset[idx]
        total_len = len(sample["dynamic"])
        assert total_len <= cfg.max_seq_len, (
            f"Sample {idx} total length {total_len} exceeds max_seq_len={cfg.max_seq_len} "
            f"under batch_mode={batch_mode}, static_inclusion_mode=prepend"
        )

    batch = dataset.collate([dataset[i] for i in range(len(dataset))])
    seq_axis = batch.time_delta_days.shape[-1]
    assert seq_axis <= cfg.max_seq_len, (
        f"Collated batch shape {batch.time_delta_days.shape} exceeds max_seq_len={cfg.max_seq_len}"
    )


def test_step_through_prepend_exhausts_window(tensorized_MEDS_dataset):
    """PREPEND + max_seq_len == len(static_code) leaves zero effective dynamic window.

    The fixture subjects all have two static codes; `max_seq_len=2` in SM + PREPEND
    therefore reserves every slot for static data, leaving no room for dynamic events.
    The dataset raises on `__init__` rather than silently skipping the subjects.
    """
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=2,
        seq_sampling_strategy="step_through",
        step_through_stride=1,
        batch_mode="SM",
        static_inclusion_mode="prepend",
    )
    with pytest.raises(ValueError, match=r"Effective dynamic window size"):
        MEDSPytorchDataset(cfg, split="train")


def test_step_through_stride_exceeds_effective_window(tensorized_MEDS_dataset):
    """`step_through_stride` greater than the per-subject effective window must raise.

    The stride is the distance between successive window *starts*; if it's wider than
    the window itself, we leave uncovered gaps between windows — step-through is
    supposed to guarantee full coverage, so this is rejected at `__init__` time.
    """
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=3,
        seq_sampling_strategy="step_through",
        step_through_stride=10,  # >> max_seq_len = effective_window in INCLUDE mode
        batch_mode="SM",
        static_inclusion_mode="include",
    )
    with pytest.raises(ValueError, match=r"step_through stride .* exceeds the effective window width"):
        MEDSPytorchDataset(cfg, split="train")


def test_step_through_ends_sm_static_only_subject(tensorized_MEDS_dataset):
    """`_step_through_ends_sm` emits a single trivial window for static-only subjects.

    Tokenization's full-outer join surfaces `null` in `measurements_per_event` for a
    subject that appears only in static data. The SM step-through walker must handle
    this without crashing — it emits `([0], [0])` so `self.index` still has one entry
    per subject and downstream `__getitem__` returns an empty dynamic sequence.
    """
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=5,
        seq_sampling_strategy="step_through",
        step_through_stride=3,
        batch_mode="SM",
        static_inclusion_mode="omit",
    )
    dataset = MEDSPytorchDataset(cfg, split="train")

    # Patch one subject's `measurements_per_event` to null (simulating a static-only
    # subject) and re-run `_step_through_ends_sm` for it. polars DataFrames are
    # immutable; replace the whole shard df with a version where subject 0's
    # measurements_per_event is None.
    subject_id = next(iter(dataset.subj_locations))
    shard, subj_idx = dataset.subj_locations[subject_id]
    original_df = dataset.schema_dfs_by_shard[shard]
    n_rows = len(original_df)
    null_col = pl.Series(
        "measurements_per_event",
        [None if i == subj_idx else original_df["measurements_per_event"][i] for i in range(n_rows)],
        dtype=original_df.schema["measurements_per_event"],
    )
    dataset.schema_dfs_by_shard[shard] = original_df.with_columns(null_col)

    assert dataset._step_through_ends_sm(subject_id, stride=3, effective_window=5) == ([0], [0])
