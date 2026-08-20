from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import torch

pytestmark = pytest.mark.lightning

# `pytest.importorskip` at module scope skips the whole file when the `lightning` extra
# isn't installed — avoids an `if _HAS_LIGHTNING:` block wrapping every test.
lightning = pytest.importorskip("lightning")
LightningDataModule = lightning.LightningDataModule

if TYPE_CHECKING:
    # `Datamodule` is only used as a type annotation below; with
    # `from __future__ import annotations` those become strings at runtime, so a
    # TYPE_CHECKING-only import avoids the `E402` noqa we'd otherwise need (the import
    # has to come after `pytest.importorskip` so the skip fires before the extension is
    # touched) and keeps the module import-order clean.
    from meds_torchdata.extensions import Datamodule


def test_lightning_datamodule(sample_lightning_datamodule: Datamodule):
    assert isinstance(sample_lightning_datamodule, LightningDataModule)

    try:
        sample_lightning_datamodule.train_dataloader()
        sample_lightning_datamodule.val_dataloader()
        sample_lightning_datamodule.test_dataloader()
    except Exception as e:
        raise AssertionError(f"Failed to create dataloaders: {e}") from e


def test_seeded_datamodules_reproduce_shuffle_order_and_propagate_epoch(sample_dataset_config):
    from meds_torchdata.extensions import Datamodule

    cfg = dataclasses.replace(sample_dataset_config, subsequence_sampling_seed=42)
    first = Datamodule(config=cfg, batch_size=2)
    second = Datamodule(config=dataclasses.replace(cfg), batch_size=2)

    first_order = torch.randperm(len(first.train_dataset), generator=first.train_dataloader_generator)
    second_order = torch.randperm(len(second.train_dataset), generator=second.train_dataloader_generator)
    torch.testing.assert_close(first_order, second_order)

    first.set_epoch(3)
    assert first.train_dataset.epoch == 3

    third = Datamodule(config=dataclasses.replace(cfg), batch_size=2)
    third.set_epoch(3)
    assert third.train_dataset.epoch == 3


def test_lightning_datamodule_with_task(
    sample_lightning_datamodule_with_task: Datamodule,
):
    assert isinstance(sample_lightning_datamodule_with_task, LightningDataModule)

    try:
        sample_lightning_datamodule_with_task.train_dataloader()
        sample_lightning_datamodule_with_task.val_dataloader()
        sample_lightning_datamodule_with_task.test_dataloader()
    except Exception as e:
        raise AssertionError(f"Failed to create dataloaders: {e}") from e

    sample_batch = next(iter(sample_lightning_datamodule_with_task.train_dataloader()))
    assert sample_batch.boolean_value is not None


def test_lightning_datamodule_with_index(
    sample_lightning_datamodule_with_index: Datamodule,
):
    assert isinstance(sample_lightning_datamodule_with_index, LightningDataModule)

    try:
        sample_lightning_datamodule_with_index.train_dataloader()
        sample_lightning_datamodule_with_index.val_dataloader()
        sample_lightning_datamodule_with_index.test_dataloader()
    except Exception as e:
        raise AssertionError(f"Failed to create dataloaders: {e}") from e

    sample_batch = next(iter(sample_lightning_datamodule_with_index.train_dataloader()))
    assert sample_batch.boolean_value is None
