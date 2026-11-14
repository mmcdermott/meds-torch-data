"""This file exports pytest fixtures for static pytorch datasets for downstream testing use."""

import shlex
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest
from meds import LabelSchema
from meds_testing_helpers.dataset import MEDSDataset

from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
from meds_torchdata.extensions import _HAS_LIGHTNING


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset(simple_static_MEDS: Path) -> Path:
    with tempfile.TemporaryDirectory() as cohort_dir:
        cohort_dir = Path(cohort_dir)

        command = [
            "MTD_preprocess",
            f"MEDS_dataset_dir={simple_static_MEDS!s}",
            f"output_dir={cohort_dir!s}",
            # The synthetic data records birth with a `DOB` code rather than the MEDS-standard
            # `MEDS_BIRTH`. Quoted twice: once for Hydra's override grammar, once for the shell.
            shlex.quote("birth_code_regex='^DOB$'"),
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        yield cohort_dir


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset_with_task(
    tensorized_MEDS_dataset: Path,
    simple_static_MEDS_dataset_with_task: Path,
) -> tuple[Path, Path, str]:
    cohort_dir = tensorized_MEDS_dataset

    D = MEDSDataset(root_dir=simple_static_MEDS_dataset_with_task)

    if len(D.task_names) != 1:  # pragma: no cover
        raise ValueError("Expected only one task in the dataset.")

    return cohort_dir, D.task_root_dir, D.task_names[0]


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset_with_index(
    tensorized_MEDS_dataset: Path,
    simple_static_MEDS_dataset_with_task: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, str]:
    cohort_dir = tensorized_MEDS_dataset

    D = MEDSDataset(root_dir=simple_static_MEDS_dataset_with_task)

    if len(D.task_names) != 1:  # pragma: no cover
        raise ValueError("Expected only one task in the dataset.")

    new_root = tmp_path_factory.mktemp("tensorized_MEDS_dataset_with_index")

    task_name = D.task_names[0]
    index_name = "task_index_no_labels"

    task_dir = D.task_root_dir / task_name
    for fp in task_dir.rglob("*.parquet"):
        relative_path = fp.relative_to(task_dir)
        out_fp = new_root / index_name / relative_path
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        df = pl.read_parquet(
            fp,
            columns=[LabelSchema.subject_id_name, LabelSchema.prediction_time_name],
            use_pyarrow=True,
        )
        df.write_parquet(out_fp, use_pyarrow=True)

    return cohort_dir, new_root, index_name


@pytest.fixture(scope="session")
def sample_dataset_config(tensorized_MEDS_dataset: Path) -> MEDSTorchDataConfig:
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=10,
    )


@pytest.fixture(scope="session")
def sample_dataset_config_with_task(
    tensorized_MEDS_dataset_with_task: tuple[Path, Path, str],
) -> MEDSTorchDataConfig:
    cohort_dir, tasks_dir, task_name = tensorized_MEDS_dataset_with_task

    return MEDSTorchDataConfig(
        tensorized_cohort_dir=cohort_dir,
        task_labels_dir=(tasks_dir / task_name),
        max_seq_len=10,
        seq_sampling_strategy="to_end",
    )


@pytest.fixture(scope="session")
def sample_dataset_config_with_index(
    tensorized_MEDS_dataset_with_index: tuple[Path, Path, str],
) -> MEDSTorchDataConfig:
    cohort_dir, tasks_dir, task_name = tensorized_MEDS_dataset_with_index

    return MEDSTorchDataConfig(
        tensorized_cohort_dir=cohort_dir,
        task_labels_dir=(tasks_dir / task_name),
        max_seq_len=10,
        seq_sampling_strategy="to_end",
    )


@pytest.fixture(scope="session")
def sample_pytorch_dataset(sample_dataset_config: MEDSTorchDataConfig) -> MEDSPytorchDataset:
    return MEDSPytorchDataset(sample_dataset_config, split="train")


@pytest.fixture(scope="session")
def sample_pytorch_dataset_with_task(
    sample_dataset_config_with_task: MEDSTorchDataConfig,
) -> MEDSPytorchDataset:
    return MEDSPytorchDataset(sample_dataset_config_with_task, split="train")


@pytest.fixture(scope="session")
def sample_pytorch_dataset_with_index(
    sample_dataset_config_with_index: MEDSTorchDataConfig,
) -> MEDSPytorchDataset:
    return MEDSPytorchDataset(sample_dataset_config_with_index, split="train")


if _HAS_LIGHTNING:
    from meds_torchdata.extensions import Datamodule

    @pytest.fixture(scope="session")
    def sample_lightning_datamodule(sample_dataset_config: MEDSTorchDataConfig) -> Datamodule:
        return Datamodule(config=sample_dataset_config, batch_size=2)

    @pytest.fixture(scope="session")
    def sample_lightning_datamodule_with_task(
        sample_dataset_config_with_task: MEDSTorchDataConfig,
    ) -> Datamodule:
        return Datamodule(config=sample_dataset_config_with_task, batch_size=2)

    @pytest.fixture(scope="session")
    def sample_lightning_datamodule_with_index(
        sample_dataset_config_with_index: MEDSTorchDataConfig,
    ) -> Datamodule:
        return Datamodule(config=sample_dataset_config_with_index, batch_size=2)
