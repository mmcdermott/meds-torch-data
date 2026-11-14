"""Lightning-level tests that random subsequence sampling varies across training epochs."""

from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = pytest.mark.lightning

lightning = pytest.importorskip("lightning")


class _RecordingModule(lightning.LightningModule):
    """Minimal module that records the sampled code rows it sees in each training epoch."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.rows_by_epoch: dict[int, list[tuple[int, ...]]] = {}

    def training_step(self, batch, batch_idx):
        rows = [tuple(row.tolist()) for row in batch.code]
        self.rows_by_epoch.setdefault(self.current_epoch, []).extend(rows)
        return (self.weight**2).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.mark.parametrize("seq_sampling_strategy", ["random", "balanced_random"])
def test_random_subsequences_vary_across_trainer_epochs(sample_dataset_config, seq_sampling_strategy):
    """A plain `trainer.fit` must not replay identical random subsequences every epoch.

    Random subsequence sampling exists to expose the model to different windows of long records across epochs;
    this guards against any change (e.g. epoch-keyed seeding) that would silently freeze the sampled windows
    under the normal Lightning training loop.
    """
    from meds_torchdata.extensions import Datamodule

    cfg = dataclasses.replace(
        sample_dataset_config, max_seq_len=3, seq_sampling_strategy=seq_sampling_strategy
    )
    datamodule = Datamodule(config=cfg, batch_size=2)
    module = _RecordingModule()

    n_epochs = 5
    trainer = lightning.Trainer(
        max_epochs=n_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
    )
    trainer.fit(module, datamodule=datamodule)

    # Compare the multiset of sampled windows per epoch, so shuffle order does not matter.
    per_epoch = [sorted(module.rows_by_epoch[e]) for e in range(n_epochs)]
    assert len({tuple(rows) for rows in per_epoch}) > 1, (
        f"Every one of {n_epochs} epochs sampled exactly the same subsequences: {per_epoch[0]}"
    )
