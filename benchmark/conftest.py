"""Test set-up and fixtures code."""


import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

PREPROCESS_SCRIPT = "MTD_preprocess"


def pytest_addoption(parser):
    parser.addoption(
        "--persistent_cache_dir",
        default=None,
        help="Use this local (preserved) directory for persistent caching. Directory must exist.",
    )


@contextmanager
def _test_dir(request):
    if request.config.getoption("--persistent_cache_dir"):
        cache_dir = Path(request.config.getoption("--persistent_cache_dir"))
        if not cache_dir.exists():
            raise FileNotFoundError("Persistent cache directory does not exist.")
        else:
            yield cache_dir
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset(request, generated_sample_MEDS: Path) -> tuple[Path, Path]:
    with _test_dir(request) as cohort_dir:
        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={str(generated_sample_MEDS)}",
            f"output_dir={cohort_dir}",
            "do_reshard=True",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        yield generated_sample_MEDS, Path(cohort_dir)
