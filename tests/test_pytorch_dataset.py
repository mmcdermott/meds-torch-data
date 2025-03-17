from meds_torchdata.pytorch_dataset import MEDSPytorchDataset


def test_dataset(sample_pytorch_dataset: MEDSPytorchDataset):
    pyd = sample_pytorch_dataset

    assert len(pyd) == 4, "The dataset should have 4 samples corresponding to the train subjects."
    assert set(pyd.subject_ids) == {239684, 1195293, 68729, 814703}

    samps = []
    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"
        samps.append(samp)

    full_batch = pyd.collate(samps)
    assert full_batch is not None
    assert "code" in full_batch, "The batch should have the code sequence."
    assert "mask" in full_batch, "The batch should have the mask sequence."
    assert "numeric_value_mask" in full_batch, "The batch should have the numeric value mask."
    assert "time_delta_days" in full_batch, "The batch should have the time delta days."
    assert "numeric_value" in full_batch, "The batch should have the numeric value."

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert batch is not None


def test_dataset_with_task(sample_pytorch_dataset_with_task: MEDSPytorchDataset):
    pyd = sample_pytorch_dataset_with_task

    assert len(pyd) == 13, "The dataset should have 10 task samples corresponding to the train samples."
    assert pyd.index == [
        (239684, 0, 3),
        (239684, 0, 4),
        (239684, 0, 5),
        (1195293, 0, 3),
        (1195293, 0, 4),
        (1195293, 0, 6),
        (68729, 0, 2),
        (68729, 0, 2),
        (68729, 0, 2),
        (68729, 0, 2),
        (814703, 0, 2),
        (814703, 0, 2),
        (814703, 0, 2),
    ]

    samps = []
    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"
        assert "boolean_value" in samp, "Each sample in the labeled setting should have the label"
        samps.append(samp)

    full_batch = pyd.collate(samps)
    assert full_batch is not None
    assert "code" in full_batch, "The batch should have the code sequence."
    assert "mask" in full_batch, "The batch should have the mask sequence."
    assert "numeric_value_mask" in full_batch, "The batch should have the numeric value mask."
    assert "time_delta_days" in full_batch, "The batch should have the time delta days."
    assert "numeric_value" in full_batch, "The batch should have the numeric value."
    assert "boolean_value" in full_batch, "The batch should have the label in the labeled setting."

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert batch is not None
    assert "boolean_value" in batch, "The batch should have the label in the labeled setting."
