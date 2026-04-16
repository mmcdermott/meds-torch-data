"""Automated stage tests driven by `@Stage.register(examples_dir=...)`.

Each stage's registered example scenarios are materialized into a fresh temp dir, the stage
is run via `MEDS_transform-stage`, and the outputs are validated against the scenario's
declared expectations. The `stage_example` fixture and its parametrization come from
`MEDS_transforms.pytest_plugin`, enabled in `conftest.py`.
"""


def test_stage_example(stage_example):
    stage_example.test()
