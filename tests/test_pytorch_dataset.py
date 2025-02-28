from pathlib import Path

from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.pytorch_dataset import MEDSPytorchDataset


def test_dataset(tensorized_MEDS_dataset: tuple[Path, Path]):
    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset[1],
        max_seq_len=10,
    )

    pyd = MEDSPytorchDataset(config, split="train")

    assert len(pyd) == 4, "The dataset should have 4 samples corresponding to the train subjects."

    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert batch is not None
