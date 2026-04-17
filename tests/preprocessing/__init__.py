"""Shared constants for the preprocessing CLI tests.

Stage-level testing lives in `test_stages.py` (declarative YAML scenarios discovered via
`@Stage.register(examples_dir=..., example_class=MTDStageExample)` and parametrized by
`MEDS_transforms.pytest_plugin`'s `stage_example` fixture). This module only exposes the
one constant that `test_preprocess.py`'s CLI-level tests still reference.
"""

PREPROCESS_SCRIPT = "MTD_preprocess"
