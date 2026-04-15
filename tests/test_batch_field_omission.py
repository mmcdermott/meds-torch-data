"""Tests for the dynamic-batch-field omission flags (#46, #47).

`MEDSTorchDataConfig.include_numeric_value=False` drops `numeric_value` and
`numeric_value_mask` from the collated `MEDSTorchBatch`; `include_time_delta=False` drops
`time_delta_days`. The `code` tensor is always present (it defines the batch mode and
shape), and `event_mask` stays around in SEM mode (it's mode-critical).

Most of the config-level validation is covered by the `MEDSTorchDataConfig` class doctest;
this file only checks the end-to-end collate semantics, which a doctest can't express
cleanly because it iterates the sample dataset fixture.
"""

import pytest

from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig


@pytest.mark.parametrize("batch_mode", ["SEM", "SM"])
@pytest.mark.parametrize(
    ("include_numeric_value", "include_time_delta"),
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_collate_honors_omission_flags(
    tensorized_MEDS_dataset, batch_mode, include_numeric_value, include_time_delta
):
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=5,
        seq_sampling_strategy="to_end",
        batch_mode=batch_mode,
        include_numeric_value=include_numeric_value,
        include_time_delta=include_time_delta,
    )
    dataset = MEDSPytorchDataset(cfg, split="train")
    batch = dataset.collate([dataset[i] for i in range(len(dataset))])

    # code is always present.
    assert batch.code is not None

    # Event mask is mode-driven, not omitted by these flags.
    if batch_mode == "SEM":
        assert batch.event_mask is not None

    if include_numeric_value:
        assert batch.numeric_value is not None
        assert batch.numeric_value_mask is not None
    else:
        assert batch.numeric_value is None
        assert batch.numeric_value_mask is None

    if include_time_delta:
        assert batch.time_delta_days is not None
    else:
        assert batch.time_delta_days is None


def test_omission_flags_compose_with_prepend(tensorized_MEDS_dataset):
    """Dropping numeric / time-delta must still work with `static_inclusion_mode=PREPEND`.

    PREPEND in SM mode used to read `out["time_delta_days"].shape[1]` to size the static mask; that code path
    is now routed through `out["code"]` so the mask still builds correctly when `include_time_delta=False`.
    """
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=8,
        seq_sampling_strategy="to_end",
        batch_mode="SM",
        static_inclusion_mode="prepend",
        include_numeric_value=False,
        include_time_delta=False,
    )
    dataset = MEDSPytorchDataset(cfg, split="train")
    batch = dataset.collate([dataset[i] for i in range(len(dataset))])

    assert batch.numeric_value is None
    assert batch.time_delta_days is None
    assert batch.static_mask is not None
    # static_mask should have the same `seq_len` axis as code in SM mode.
    assert batch.static_mask.shape == batch.code.shape
