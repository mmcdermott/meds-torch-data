import dataclasses
from functools import cached_property
from typing import Generic, TypeVar, cast

import lightning as L
from hydra.utils import get_class
from meds import held_out_split, train_split, tuning_split
from torch.utils.data import DataLoader

from ..config import MEDSTorchDataConfig
from ..pytorch_dataset import MEDSPytorchDataset

DatasetT = TypeVar("DatasetT", bound=MEDSPytorchDataset)


class Datamodule(L.LightningDataModule, Generic[DatasetT]):
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
        >>> isinstance(D.train_dataset, MEDSPytorchDataset)
        True
        >>> D.train_dataset.split
        'train'
        >>> isinstance(D.val_dataset, MEDSPytorchDataset)
        True
        >>> D.val_dataset.split
        'tuning'
        >>> isinstance(D.test_dataset, MEDSPytorchDataset)
        True
        >>> D.test_dataset.split
        'held_out'

    After construction, we can access dataloaders for training, validation, and testing. The train dataloader
    shuffles so doesn't return stable outputs, but the others do not shuffle.

        >>> L.seed_everything(0)
        0
        >>> train_dataloader = D.train_dataloader()
        >>> next(iter(train_dataloader))
        MEDSTorchBatch(code=tensor([[...]]), ..., boolean_value=None)
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

    You can also set the pin_memory flag to True, and it will be applied to the created dataloaders.

        >>> D = Datamodule(config=sample_dataset_config, batch_size=1, pin_memory=True)
        >>> D.shared_dataloader_kwargs
        {'batch_size': 1, 'pin_memory': True}
        >>> test_dataloader = D.test_dataloader()
        >>> next(iter(test_dataloader))
        MEDSTorchBatch(code=tensor([[ 5,  2, 10, 11, 10, 11, 10, 11,  4]]), ..., boolean_value=None)

    You can also override the dataset class used by the datamodule via the `data_class` argument. This is
    useful for injecting convenience helpers or light instrumentation while retaining the core
    `MEDSPytorchDataset` functionality.

        >>> class MyDataset(MEDSPytorchDataset):
        ...     def who_am_i(self):
        ...         return "custom"
        >>> D = Datamodule(config=sample_dataset_config, data_class=MyDataset, batch_size=1)
        >>> isinstance(D.train_dataset, MyDataset)
        True
        >>> isinstance(D.val_dataset, MyDataset)
        True
        >>> isinstance(D.test_dataset, MyDataset)
        True
        >>> D.train_dataset.who_am_i()
        'custom'
        >>> D.train_dataset is D.val_dataset
        False
    """

    config: MEDSTorchDataConfig
    data_class = MEDSPytorchDataset
    batch_size: int
    num_workers: int | None
    pin_memory: bool | None = None

    def __init__(
        self,
        config: MEDSTorchDataConfig,
        data_class: type[DatasetT] | str = MEDSPytorchDataset,
        batch_size: int = 32,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
    ):
        super().__init__()
        self.config = config
        if isinstance(data_class, str):
            data_class = cast("type[DatasetT]", get_class(data_class))
        self.data_class: type[DatasetT] = cast("type[DatasetT]", data_class)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.save_hyperparameters(
            {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "config": dataclasses.asdict(config),
            }
        )

    @property
    def shared_dataloader_kwargs(self) -> dict:
        out = {"batch_size": self.batch_size}
        for param in {"num_workers", "pin_memory"}:
            if getattr(self, param) is not None:
                out[param] = getattr(self, param)
        return out

    @cached_property
    def train_dataset(self) -> DatasetT:
        return self.data_class(self.config, split=train_split)

    @cached_property
    def val_dataset(self) -> DatasetT:
        return self.data_class(self.config, split=tuning_split)

    @cached_property
    def test_dataset(self) -> DatasetT:
        return self.data_class(self.config, split=held_out_split)

    def __dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, collate_fn=dataset.collate, **self.shared_dataloader_kwargs, **kwargs)

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False)
