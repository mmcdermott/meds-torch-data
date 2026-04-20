import pytest

pytestmark = pytest.mark.lightning

# `pytest.importorskip` at module scope skips the whole file when the `lightning` extra
# isn't installed — avoids an `if _HAS_LIGHTNING:` block wrapping every test.
lightning = pytest.importorskip("lightning")
LightningDataModule = lightning.LightningDataModule

from meds_torchdata.extensions import Datamodule  # noqa: E402


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
