"""Test set-up and fixtures code."""


import subprocess
import tempfile
from pathlib import Path

import pytest

from .preprocessing import PREPROCESS_SCRIPT


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset(simple_static_MEDS: Path) -> tuple[Path, Path]:
    with tempfile.TemporaryDirectory() as cohort_dir:
        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={str(simple_static_MEDS)}",
            f"output_dir={cohort_dir}",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        yield simple_static_MEDS, Path(cohort_dir)
