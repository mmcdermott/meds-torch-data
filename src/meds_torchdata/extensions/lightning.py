from functools import cached_property

import lightning as L
from meds import held_out_split, train_split, tuning_split
from torch.utils.data import DataLoader

from ..config import MEDSTorchDataConfig
from ..pytorch_dataset import MEDSPytorchDataset


class Datamodule(L.LightningDataModule):
    """A lightning datamodule for a MEDSPytorchDataset.

    > [!NOTE]
    > This class does not do any intelligent preparation of the dataset; it merely packages a pre-processed
    > dataset in the lighting data module format.

    > [!NOTE]
    > Lightning uses a different convention for the split names than MEDS. Namely, it uses "train", "val", and
    > "test", while MEDS uses "train", "tuning", and "held-out". This class maps the MEDS split names to the
    > Lightning split names as follows: "train" -> "train", "tuning" -> "val", and "held-out" -> "test".

    Attributes:
        config: The configuration for the dataset.
        batch_size: The batch size for the dataloaders. Defaults to 32.
        num_workers: The number of workers for the dataloaders. Defaults to 0.

    Examples:
        >>> D = Datamodule(config=sample_dataset_config, batch_size=2)

    After construction, we can access dataloaders for training, validation, and testing. The train dataloader
    shuffles so doesn't return stable outputs, but the others do not shuffle.

        >>> L.seed_everything(0)
        0
        >>> train_dataloader = D.train_dataloader()
        >>> next(iter(train_dataloader))
        MEDSTorchBatch(code=tensor([[ 5,  1, 10, 11, 10, 11, 10, 11, 10, 11],
                                    [ 5,  2, 10, 11,  4,  0,  0,  0,  0,  0]]),
                       ...,
                       boolean_value=None)
        >>> val_dataloader = D.val_dataloader()
        >>> next(iter(val_dataloader))
        MEDSTorchBatch(code=tensor([[ 5,  3, 10, 11,  4]]), ..., boolean_value=None)

    You can also set the number of workers to a non-zero value, and it will be applied to the created
    dataloaders, along with batch size, through the `shared_dataloader_kwargs` property.

        >>> D = Datamodule(config=sample_dataset_config, batch_size=1, num_workers=4)
        >>> D.shared_dataloader_kwargs
        {'batch_size': 1, 'num_workers': 4}
        >>> test_dataloader = D.test_dataloader()
        >>> next(iter(test_dataloader))
        MEDSTorchBatch(code=tensor([[ 5,  2, 10, 11, 10, 11, 10, 11,  4]]), ..., boolean_value=None)
    """

    config: MEDSTorchDataConfig
    batch_size: int
    num_workers: int | None

    def __init__(
        self,
        config: MEDSTorchDataConfig,
        batch_size: int = 32,
        num_workers: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def shared_dataloader_kwargs(self) -> dict:
        out = {"batch_size": self.batch_size}
        if self.num_workers is not None:
            out["num_workers"] = self.num_workers
        return out

    @cached_property
    def train_dataset(self) -> MEDSPytorchDataset:
        return MEDSPytorchDataset(self.config, split=train_split)

    @cached_property
    def val_dataset(self) -> MEDSPytorchDataset:
        return MEDSPytorchDataset(self.config, split=tuning_split)

    @cached_property
    def test_dataset(self) -> MEDSPytorchDataset:
        return MEDSPytorchDataset(self.config, split=held_out_split)

    def __dataloader(self, dataset: MEDSPytorchDataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, collate_fn=dataset.collate, **self.shared_dataloader_kwargs, **kwargs)

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False)
