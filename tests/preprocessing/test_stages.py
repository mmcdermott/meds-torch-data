"""Automated stage tests driven by registered `StageExample` scenarios.

Each stage's registered example scenarios are materialized into a fresh temp dir, the stage
is run via `MEDS_transform-stage`, and the outputs are validated against the scenario's
declared expectations. The `stage_example` fixture and its parametrization come from
`MEDS_transforms.pytest_plugin`, which is auto-loaded via the `meds-transforms` package's
`pytest11` entry point. Our stage modules live at
`preprocessing/<stage>/<stage>.py` with an adjacent `<stage>/examples/` subdir, which lets
`Stage.register` auto-infer `examples_dir` without an explicit argument.

The pipeline test runs tokenization + tensorization end-to-end via `MEDS_transform-pipeline`
and validates each stage's output against its registered `out_data.yaml`. We scope it to our
two stages (rather than the full 5-stage MTD pipeline) because the upstream pre-tokenization
stages have their own chained test fixtures and hooking all five into a single chained
scenario would pull in substantial upstream fixture machinery for no additional coverage —
our stages' singleton scenarios are already aligned to chain with each other.
"""

import pytest
from MEDS_transforms.pytest_plugin import pipeline_tester


def test_stage_example(stage_example):
    stage_example.test()


_PIPELINE_YAML = "input_dir: {input_dir}\noutput_dir: {output_dir}\nstages: [tokenization, tensorization]\n"


def test_pipeline_serial():
    """Chain tokenization → tensorization through `MEDS_transform-pipeline` in serial mode."""
    pipeline_tester(
        pipeline_yaml=_PIPELINE_YAML,
        stage_runner_yaml=None,
        stage_scenario_sequence=["tokenization/default", "tensorization/default"],
    )


@pytest.mark.parallelized
def test_pipeline_parallel():
    """Same pipeline, run through joblib — replaces bespoke `test_preprocess_parallel.py`.

    Gated on the `parallelized` marker so CI skips when `hydra-joblib-launcher` isn't
    installed (the existing optional dep used for the old parallel test).
    """
    pipeline_tester(
        pipeline_yaml=_PIPELINE_YAML,
        stage_runner_yaml="parallelize:\n  n_workers: 2\n  launcher: joblib\n",
        stage_scenario_sequence=["tokenization/default", "tensorization/default"],
    )
