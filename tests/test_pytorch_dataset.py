import dataclasses

import numpy as np
import pytest
import torch
from meds import DataSchema

from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig


def sampled_codes(dataset: MEDSPytorchDataset, idx: int) -> np.ndarray:
    """Return sampled dynamic codes for one dataset element."""
    return dataset[idx]["dynamic"].to_dense()["code"]


def dataloader_codes(loader: torch.utils.data.DataLoader) -> list[torch.Tensor]:
    """Return code tensors in DataLoader order."""
    return [batch.code for batch in loader]


def assert_code_batches_equal(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> None:
    """Assert two sequences of potentially ragged single-sample batches are equal."""
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_batch, expected_batch)


def test_dataset(sample_pytorch_dataset: MEDSPytorchDataset):
    pyd = sample_pytorch_dataset

    assert len(pyd) == 4, "The dataset should have 4 samples corresponding to the train subjects."

    samps = []
    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"
        samps.append(samp)

    full_batch = pyd.collate(samps)
    assert isinstance(full_batch, MEDSTorchBatch)

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert isinstance(batch, MEDSTorchBatch)


def test_seeded_random_sampling_is_reproducible_by_epoch_and_index(sample_dataset_config):
    cfg = dataclasses.replace(
        sample_dataset_config,
        max_seq_len=3,
        seq_sampling_strategy="random",
        subsequence_sampling_seed=42,
    )
    first = MEDSPytorchDataset(cfg, split="train")
    second = MEDSPytorchDataset(dataclasses.replace(cfg), split="train")

    epoch_zero = sampled_codes(first, 0)
    np.testing.assert_array_equal(epoch_zero, sampled_codes(first, 0))
    np.testing.assert_array_equal(epoch_zero, sampled_codes(second, 0))

    first.set_epoch(1)
    second.set_epoch(1)
    epoch_one = sampled_codes(first, 0)
    np.testing.assert_array_equal(epoch_one, sampled_codes(second, 0))
    assert not np.array_equal(epoch_zero, epoch_one)

    different_seed = MEDSPytorchDataset(dataclasses.replace(cfg, subsequence_sampling_seed=43), split="train")
    assert not np.array_equal(epoch_zero, sampled_codes(different_seed, 0))


def test_seeded_sampling_is_independent_of_dataloader_workers(sample_dataset_config):
    cfg = dataclasses.replace(
        sample_dataset_config,
        max_seq_len=3,
        seq_sampling_strategy="balanced_random",
        subsequence_sampling_seed=42,
    )
    single_worker_dataset = MEDSPytorchDataset(cfg, split="train")
    multi_worker_dataset = MEDSPytorchDataset(dataclasses.replace(cfg), split="train")

    single_worker_loader = single_worker_dataset.get_dataloader(batch_size=1, shuffle=False, num_workers=0)
    multi_worker_loader = multi_worker_dataset.get_dataloader(
        batch_size=1, shuffle=False, num_workers=2, persistent_workers=True
    )

    expected = dataloader_codes(single_worker_loader)
    actual = dataloader_codes(multi_worker_loader)

    assert_code_batches_equal(actual, expected)

    single_worker_dataset.set_epoch(1)
    multi_worker_dataset.set_epoch(1)
    expected = dataloader_codes(single_worker_loader)
    actual = dataloader_codes(multi_worker_loader)

    assert_code_batches_equal(actual, expected)


@pytest.mark.parametrize("epoch", [-1, 1.5, True, 2**63])
def test_set_epoch_rejects_invalid_values(sample_pytorch_dataset, epoch):
    with pytest.raises(ValueError, match=r"epoch must be an integer in \[0, 2\*\*63\)"):
        sample_pytorch_dataset.set_epoch(epoch)


def test_schema_df_last_time(tensorized_MEDS_dataset_with_index):
    cohort_dir, tasks_dir, task_name = tensorized_MEDS_dataset_with_index
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=cohort_dir,
        task_labels_dir=(tasks_dir / task_name),
        max_seq_len=10,
        seq_sampling_strategy="to_end",
        include_window_last_observed_in_schema=True,
    )

    pyd = MEDSPytorchDataset(cfg, split="train")
    assert MEDSPytorchDataset.LAST_TIME in pyd.schema_df.columns

    subj, end_idx = pyd.index[0]
    shard, subj_idx = pyd.subj_locations[subj]
    times = pyd.schema_dfs_by_shard[shard][DataSchema.time_name][subj_idx]
    assert pyd.schema_df[pyd.LAST_TIME][0] == times[end_idx - 1]


def test_dataset_with_task(sample_pytorch_dataset_with_task: MEDSPytorchDataset):
    pyd = sample_pytorch_dataset_with_task

    assert len(pyd) == 13, "The dataset should have 10 task samples corresponding to the train samples."
    assert pyd.index == [
        (239684, 3),
        (239684, 4),
        (239684, 5),
        (1195293, 3),
        (1195293, 4),
        (1195293, 6),
        (68729, 2),
        (68729, 2),
        (68729, 2),
        (68729, 2),
        (814703, 2),
        (814703, 2),
        (814703, 2),
    ]

    samps = []
    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"
        assert "boolean_value" in samp, "Each sample in the labeled setting should have the label"
        samps.append(samp)

    full_batch = pyd.collate(samps)
    assert isinstance(full_batch, MEDSTorchBatch)

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert isinstance(batch, MEDSTorchBatch)
