from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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
