<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo_light.svg">
    <img width="520" height="200" alt="MEDS Logot" src="static/logo_light.svg">
  </picture>
</p>

# MEDS TorchData: A PyTorch Dataset Class for MEDS Datasets

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Python 3.11+](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![PyPI - Version](https://img.shields.io/pypi/v/meds-torch-data)](https://pypi.org/project/meds-torch-data/)
[![MEDS v0.4](https://img.shields.io/badge/MEDS-0.4-blue)](https://medical-event-data-standard.github.io/)
[![Documentation Status](https://readthedocs.org/projects/meds-testing-helpers/badge/?version=latest)](https://meds-testing-helpers.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/mmcdermott/meds-torch-data/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/meds-torch-data/actions/workflows/tests.yaml)
[![Test Coverage](https://codecov.io/github/mmcdermott/meds-torch-data/graph/badge.svg?token=BV119L5JQJ)](https://codecov.io/github/mmcdermott/meds-torch-data)
[![Code Quality](https://github.com/mmcdermott/meds-torch-data/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/meds-torch-data/actions/workflows/code-quality-main.yaml)
[![Hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Contributors](https://img.shields.io/github/contributors/oufattole/meds-torch.svg)](https://github.com/mmcdermott/meds-torch-data/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/meds-torch-data/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/meds-torch-data#license)

## 🚀 Quick Start

### Step 1: Install

```bash
pip install meds-torch-data
```

### Step 2: Data Tensorization

> [!WARNING]
> If your dataset is not sharded by split, you need to run a reshard to split stage first! You can enable this
> by adding the `do_reshard=True` argument to the command below.

If your input MEDS dataset lives in `$MEDS_ROOT` and you want to store your pre-processed files in
`$PYD_ROOT`, you run:

```bash
MTD_preprocess MEDS_dataset_dir="$MEDS_ROOT" output_dir="$PYD_ROOT"
```

### Step 3: Use the dataset

To use a dataset, you need to (1) define your configuration object and (2) create the dataset object. The only
required configuration parameters are `tensorized_cohort_dir`, which points to the root directory containing
the pre-processed data on disk (`$PYD_ROOT` in the above example), and `max_seq_len`, which is the maximum
sequence length you want to use for your model. Here's an example:

```python
import os
from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig

cfg = MEDSTorchDataConfig(tensorized_cohort_dir=os.environ["PYD_ROOT"], max_seq_len=512)
pyd = MEDSPytorchDataset(cfg, split="train")
```

If you want to use a specific binary classification task, you can add the `task_labels_dir` parameter to the
configuration object. This should point to a directory containing the sharded MEDS label format parquet files
for the labels. The sharding scheme is arbitrary and will not be reflected in the dataset.

That's it!

> [!NOTE]
> Only binary classification tasks are supported at this time. If you need multi-class classification or other
> kinds of tasks, please file a [GitHub issue](https://github.com/mmcdermott/meds-torch-data/issues)

## 📚 Documentation

### Design Principles

A good PyTorch dataset class should:

- Be easy to use
- Have a minimal, constant resource footprint (memory, CPU, start-up time) during model training and
    inference, _regardless of the overall dataset size_.
- Perform as much work as possible in _static, reusable dataset pre-processing_, rather than upon
    construction or in the `__getitem__` method.
- Induce effectively negligible computational overhead in the `__getitem__` method relative to model training.
- Be easily configurable, with a simple, consistent API, and cover the most common use-cases.
- Encourage efficient use of GPU resources in the resulting batches.
- Should be comprehensively documented, tested, and benchmarked for performance implications so users can
    use it reliably and effectively.

To achieve this, MEDS TorchData leverages the following design principles:

1. **Lazy Loading**: Data is loaded only when needed, and only the data needed for the current batch is
    loaded.
2. **Efficient Loading**: Data is loaded efficiently leveraging the
    [HuggingFace Safetensors](https://huggingface.co/docs/safetensors/en/index) library for raw IO through
    the nested, ragged interface encoded in the
    [Nested Ragged Tensors](https://github.com/mmcdermott/nested_ragged_tensors) library.
3. **Configurable, Transparent Pre-processing**: Mandatory data pre-processing prior to effective use in
    this library is managed through a simple
    [MEDS-Transforms](https://meds-transforms.readthedocs.io/en/latest/) pipeline which can be run on any
    MEDS dataset, after any model-specific pre-processing, via a transparent configuration file.
4. **Continuous Integration**: The library is continuously tested and benchmarked for performance
    implications, and the results are available to users.

### Examples and Detailed Usage

To see how this works, let's look at some examples. These examples will be powered by some synthetic data
defined as "fixtures" in this package's pytest stack; namely, we'll use the following fixtures:

- `simple_static_MEDS`: This will point to a Path containing a simple MEDS dataset.
- `simple_static_MEDS_dataset_with_task`: This will point to a Path containing a simple MEDS dataset
    with a boolean-value task defined. The core data is the same between both the `simple_static_MEDS` and
    this dataset, but the latter has a task defined.
- `tensorized_MEDS_dataset` fixture that points to a Path containing the tensorized and schema files for
    the `simple_static_MEDS` dataset.
- `tensorized_MEDS_dataset_with_task` fixture that points to a tuple containing:
    - A Path containing the tensorized and schema files for the `simple_static_MEDS_dataset_with_task` dataset
    - A Path pointing to the root task directory for the dataset
    - The specific task name for the dataset. Task label files will be stored in a subdir of the root task
        directory with this name.

You can find these in either the [`conftest.py`](conftest.py) file for this repository or the
[`meds_testing_helpers`](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers) package, which
this package leverages for testing.

#### Synthetic Data

To start, let's take a look at this synthetic data. It is sharded by split, and we'll look at the train split
first, which has two shards (we convert to polars just for prettier printing). It has four subjects across the
two shards:

```python
>>> import polars as pl
>>> from meds_testing_helpers.dataset import MEDSDataset
>>> D = MEDSDataset(root_dir=simple_static_MEDS)
>>> train_0 = pl.from_arrow(D.data_shards["train/0"])
>>> train_0
shape: (30, 4)
┌────────────┬─────────────────────┬────────────────────┬───────────────┐
│ subject_id ┆ time                ┆ code               ┆ numeric_value │
│ ---        ┆ ---                 ┆ ---                ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ str                ┆ f32           │
╞════════════╪═════════════════════╪════════════════════╪═══════════════╡
│ 239684     ┆ null                ┆ EYE_COLOR//BROWN   ┆ null          │
│ 239684     ┆ null                ┆ HEIGHT             ┆ 175.271118    │
│ 239684     ┆ 1980-12-28 00:00:00 ┆ DOB                ┆ null          │
│ 239684     ┆ 2010-05-11 17:41:51 ┆ ADMISSION//CARDIAC ┆ null          │
│ 239684     ┆ 2010-05-11 17:41:51 ┆ HR                 ┆ 102.599998    │
│ …          ┆ …                   ┆ …                  ┆ …             │
│ 1195293    ┆ 2010-06-20 20:24:44 ┆ HR                 ┆ 107.699997    │
│ 1195293    ┆ 2010-06-20 20:24:44 ┆ TEMP               ┆ 100.0         │
│ 1195293    ┆ 2010-06-20 20:41:33 ┆ HR                 ┆ 107.5         │
│ 1195293    ┆ 2010-06-20 20:41:33 ┆ TEMP               ┆ 100.400002    │
│ 1195293    ┆ 2010-06-20 20:50:04 ┆ DISCHARGE          ┆ null          │
└────────────┴─────────────────────┴────────────────────┴───────────────┘
>>> train_1 = pl.from_arrow(D.data_shards["train/1"])
>>> train_1
shape: (14, 4)
┌────────────┬─────────────────────┬───────────────────────┬───────────────┐
│ subject_id ┆ time                ┆ code                  ┆ numeric_value │
│ ---        ┆ ---                 ┆ ---                   ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ str                   ┆ f32           │
╞════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
│ 68729      ┆ null                ┆ EYE_COLOR//HAZEL      ┆ null          │
│ 68729      ┆ null                ┆ HEIGHT                ┆ 160.395309    │
│ 68729      ┆ 1978-03-09 00:00:00 ┆ DOB                   ┆ null          │
│ 68729      ┆ 2010-05-26 02:30:56 ┆ ADMISSION//PULMONARY  ┆ null          │
│ 68729      ┆ 2010-05-26 02:30:56 ┆ HR                    ┆ 86.0          │
│ …          ┆ …                   ┆ …                     ┆ …             │
│ 814703     ┆ 1976-03-28 00:00:00 ┆ DOB                   ┆ null          │
│ 814703     ┆ 2010-02-05 05:55:39 ┆ ADMISSION//ORTHOPEDIC ┆ null          │
│ 814703     ┆ 2010-02-05 05:55:39 ┆ HR                    ┆ 170.199997    │
│ 814703     ┆ 2010-02-05 05:55:39 ┆ TEMP                  ┆ 100.099998    │
│ 814703     ┆ 2010-02-05 07:02:30 ┆ DISCHARGE             ┆ null          │
└────────────┴─────────────────────┴───────────────────────┴───────────────┘
>>> sorted(set(train_0["subject_id"].unique()) | set(train_1["subject_id"].unique()))
[68729, 239684, 814703, 1195293]

```

#### `MEDSTorchDataConfig` Configuration Object

_Full API documentation for the configuration object can be found
[here](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/config/#meds_torchdata.config.MEDSTorchDataConfig)._

The configuration object contains two kinds of parameters: Data processing parameters and file paths.
Data processing parameters include:

- `max_seq_len`: The maximum sequence length to use for the model.
- `seq_sampling_strategy`: The strategy to use when sampling sub-sequences to return for input sequences
    longer than `max_seq_len`.
- `static_inclusion_mode`: The mode to use when including static data in the output.
- `batch_mode`: Whether to return sequences at the _measurement_ level (`"SM"`) or the _event_ level
    (`"SEM"`). Note that here, we use "_measurement_" to refer to a single row (observation) in the raw MEDS
    data, and "_event_" to refer to all measurements taken at a single time-point.
- `include_window_last_observed_in_schema`: If `True`, include the timestamp of the last observation in each
    sampled window in the dataset's `schema_df` when an index dataframe is used and the sampling strategy is
    deterministic. This functionality is useful for generative applications where the model needs to know what
    the timestamp is at the start of a generation window, for example.

Of these, `seq_sampling_strategy` and `static_inclusion_mode` are restricted, and must be of the
[`SubsequenceSamplingStrategy`](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/config/#meds_torchdata.config.SubsequenceSamplingStrategy)
and
[`StaticInclusionMode`](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/config/#meds_torchdata.config.StaticInclusionMode)
`StrEnum`s, respectively:

- `seq_sampling_strategy`: One of `["random", "to_end", "from_start"]` (defaults to `"random"`).
- `static_inclusion_mode`: One of `["include", "prepend", "omit"]` (defaults to `"include"`).

File path parameters include:

- `tensorized_cohort_dir`: The directory containing the tensorized data.
- `task_labels_dir`: The directory containing the task labels files.

It also provides a convenient property to get the vocab size for the dataset, given by the vocab indices in
the tensorized metadata. Let's start by building a configuration object for this data and inspect some of its
file-path related properties and helpers:

```python
>>> from meds_torchdata import MEDSTorchDataConfig
>>> cfg = MEDSTorchDataConfig(tensorized_MEDS_dataset, max_seq_len=5)
>>> cfg.tensorized_cohort_dir
PosixPath('/tmp/tmp...')
>>> cfg.schema_dir
PosixPath('/tmp/tmp.../tokenization/schemas')
>>> print(sorted(list(cfg.schema_fps)))
[('held_out/0', PosixPath('/tmp/tmp.../tokenization/schemas/held_out/0.parquet')),
 ('train/0', PosixPath('/tmp/tmp.../tokenization/schemas/train/0.parquet')),
 ('train/1', PosixPath('/tmp/tmp.../tokenization/schemas/train/1.parquet')),
 ('tuning/0', PosixPath('/tmp/tmp.../tokenization/schemas/tuning/0.parquet'))]
>>> print(cfg.task_labels_dir)
None
>>> print(cfg.task_labels_fps)
None
>>> print(cfg.vocab_size)
12

```

If we specify a `task_labels_dir` parameter, the config operates in task-specific mode. This allows us to use
the task-specific helpers, but it also mandates we set `seq_sampling_strategy` to `"to_end"` as you shouldn't
try to predict a downstream task without leveraging the most recent data.

```python
>>> cohort_dir, tasks_dir, task_name = tensorized_MEDS_dataset_with_task
>>> cfg = MEDSTorchDataConfig(
...     cohort_dir, max_seq_len=5, task_labels_dir=(tasks_dir / task_name)
... )
Traceback (most recent call last):
    ...
ValueError: Not sampling data till the end of the sequence when predicting for a specific task is not
permitted! This is because there is no use-case we know of where you would want to do this. If you disagree,
please let us know via a GitHub issue.
>>> cfg = MEDSTorchDataConfig(
...     cohort_dir, max_seq_len=5, task_labels_dir=(tasks_dir / task_name), seq_sampling_strategy="to_end"
... )
>>> cfg.task_labels_dir
PosixPath('/tmp/tmp.../task_labels/boolean_value_task')
>>> print(list(cfg.task_labels_fps))
[PosixPath('/tmp/tmp.../task_labels/boolean_value_task/labels_A.parquet.parquet'),
 PosixPath('/tmp/tmp.../task_labels/boolean_value_task/labels_B.parquet.parquet')]

```

Based on the `seq_sampling_strategy`, `batch_mode`, and `max_seq_len` parameters, the configuration
object also has the
[`process_dynamic_data`](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/config/#meds_torchdata.config.MEDSTorchDataConfig.process_dynamic_data)
helper function to slice the subject's dynamic data appropriately. This function is used internally, and you
will not need to use it yourself.

#### `MEDSPytorchDataset` Dataset Class

_Full API documentation for the dataset class can be found
[here](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/pytorch_dataset/#meds_torchdata.pytorch_dataset.MEDSPytorchDataset)._

Now let's build a dataset object from the synthetic data.

##### Dataset "Schema"

When we build a PyTorch dataset from it for training, with no task specified, the length will be four, as it
will correspond to each of the four subjects in the train split. The index variable contains the list of
subject IDs and the end of the allowed region of reading for the dataset. We can also see it in dataframe
format via the `schema_df`:

```python
>>> from meds_torchdata import MEDSPytorchDataset
>>> cfg = MEDSTorchDataConfig(tensorized_cohort_dir=tensorized_MEDS_dataset, max_seq_len=5)
>>> pyd = MEDSPytorchDataset(cfg, split="train")
>>> len(pyd)
4
>>> pyd.index
[(239684, 6), (1195293, 8), (68729, 3), (814703, 3)]
>>> pyd.schema_df
shape: (4, 2)
┌────────────┬─────────────────┐
│ subject_id ┆ end_event_index │
│ ---        ┆ ---             │
│ i64        ┆ u32             │
╞════════════╪═════════════════╡
│ 239684     ┆ 6               │
│ 1195293    ┆ 8               │
│ 68729      ┆ 3               │
│ 814703     ┆ 3               │
└────────────┴─────────────────┘

```

Note the index is in terms of _event indices_, not _measurement indices_ -- meaning it is the index of the
unique timestamp corresponding to the start and end of each subject's data; not the unique measurement. We can
validate that against the raw data. To do so, we'll define the simple helper function `get_event_bounds` that
will just group by the `subject_id` and `time` columns, and then calculate the event index for each subject
and show us the min and max such index, per-subject.

```python
>>> def get_event_indices(df: pl.DataFrame) -> pl.DataFrame:
...     return (
...         df
...         .group_by("subject_id", "time", maintain_order=True).agg(pl.len().alias("n_measurements"))
...         .with_row_index()
...         .select(
...             "subject_id", "time",
...             (pl.col("index") - pl.col("index").min().over("subject_id")).alias("event_idx"),
...             "n_measurements",
...         )
...     )
>>> def get_event_bounds(df: pl.DataFrame) -> pl.DataFrame:
...     return (
...         get_event_indices(df)
...         .with_columns(
...             pl.col("event_idx").max().over("subject_id").alias("max_event_idx")
...         )
...         .filter((pl.col("event_idx") == 0) | (pl.col("event_idx") == pl.col("max_event_idx")))
...         .select("subject_id", "event_idx", "time")
...     )
>>> get_event_bounds(train_0)
shape: (4, 3)
┌────────────┬───────────┬─────────────────────┐
│ subject_id ┆ event_idx ┆ time                │
│ ---        ┆ ---       ┆ ---                 │
│ i64        ┆ u32       ┆ datetime[μs]        │
╞════════════╪═══════════╪═════════════════════╡
│ 239684     ┆ 0         ┆ null                │
│ 239684     ┆ 6         ┆ 2010-05-11 19:27:19 │
│ 1195293    ┆ 0         ┆ null                │
│ 1195293    ┆ 8         ┆ 2010-06-20 20:50:04 │
└────────────┴───────────┴─────────────────────┘
>>> get_event_bounds(train_1)
shape: (4, 3)
┌────────────┬───────────┬─────────────────────┐
│ subject_id ┆ event_idx ┆ time                │
│ ---        ┆ ---       ┆ ---                 │
│ i64        ┆ u32       ┆ datetime[μs]        │
╞════════════╪═══════════╪═════════════════════╡
│ 68729      ┆ 0         ┆ null                │
│ 68729      ┆ 3         ┆ 2010-05-26 04:51:52 │
│ 814703     ┆ 0         ┆ null                │
│ 814703     ┆ 3         ┆ 2010-02-05 07:02:30 │
└────────────┴───────────┴─────────────────────┘

```

The schema changes to reflect the different split if we change the split:

```python
>>> pyd_tuning = MEDSPytorchDataset(cfg, split="tuning")
>>> pyd_tuning.schema_df
shape: (1, 2)
┌────────────┬─────────────────┐
│ subject_id ┆ end_event_index │
│ ---        ┆ ---             │
│ i64        ┆ u32             │
╞════════════╪═════════════════╡
│ 754281     ┆ 3               │
└────────────┴─────────────────┘
>>> pyd_held_out = MEDSPytorchDataset(cfg, split="held_out")
>>> pyd_held_out.schema_df
shape: (1, 2)
┌────────────┬─────────────────┐
│ subject_id ┆ end_event_index │
│ ---        ┆ ---             │
│ i64        ┆ u32             │
╞════════════╪═════════════════╡
│ 1500733    ┆ 5               │
└────────────┴─────────────────┘

```

If you use a non-existent split or have something misconfigured, you'll get an error upon Dataset creation:

```python
>>> pyd_bad = MEDSPytorchDataset(cfg, split="bad_split")
Traceback (most recent call last):
    ...
FileNotFoundError: No schema files found in /tmp/.../tokenization/schemas! If your data is not sharded by
split, this error may occur because this codebase does not handle non-split sharded data. See Issue #79 for
tracking this issue.

```

We can also inspect the schema for a dataset built with downstream task labels:

```python
>>> cohort_dir, tasks_dir, task_name = tensorized_MEDS_dataset_with_task
>>> cfg_with_task = MEDSTorchDataConfig(
...     cohort_dir, max_seq_len=5, task_labels_dir=(tasks_dir / task_name), seq_sampling_strategy="to_end"
... )
>>> pyd_with_task = MEDSPytorchDataset(cfg_with_task, split="train")
>>> pyd_with_task.schema_df
shape: (13, 4)
┌────────────┬─────────────────┬─────────────────────┬───────────────┐
│ subject_id ┆ end_event_index ┆ prediction_time     ┆ boolean_value │
│ ---        ┆ ---             ┆ ---                 ┆ ---           │
│ i64        ┆ u32             ┆ datetime[μs]        ┆ bool          │
╞════════════╪═════════════════╪═════════════════════╪═══════════════╡
│ 239684     ┆ 3               ┆ 2010-05-11 18:00:00 ┆ false         │
│ 239684     ┆ 4               ┆ 2010-05-11 18:30:00 ┆ true          │
│ 239684     ┆ 5               ┆ 2010-05-11 19:00:00 ┆ true          │
│ 1195293    ┆ 3               ┆ 2010-06-20 19:30:00 ┆ false         │
│ 1195293    ┆ 4               ┆ 2010-06-20 20:00:00 ┆ true          │
│ …          ┆ …               ┆ …                   ┆ …             │
│ 68729      ┆ 2               ┆ 2010-05-26 04:00:00 ┆ true          │
│ 68729      ┆ 2               ┆ 2010-05-26 04:30:00 ┆ true          │
│ 814703     ┆ 2               ┆ 2010-02-05 06:00:00 ┆ false         │
│ 814703     ┆ 2               ┆ 2010-02-05 06:30:00 ┆ true          │
│ 814703     ┆ 2               ┆ 2010-02-05 07:00:00 ┆ true          │
└────────────┴─────────────────┴─────────────────────┴───────────────┘

```

When we have a task or index dataframe (an index is just a task without a label), we can also ask the model to
include the last observed time in our input window in the schema, with the `include_window_last_observed_in_schema`
parameter:

```python
>>> cfg_with_end_time = MEDSTorchDataConfig(
...     cohort_dir, max_seq_len=5, task_labels_dir=(tasks_dir / task_name), seq_sampling_strategy="to_end",
...     include_window_last_observed_in_schema=True
... )
>>> pyd_with_end_time = MEDSPytorchDataset(cfg_with_end_time, split="train")
>>> pyd_with_end_time.schema_df
shape: (13, 5)
┌────────────┬─────────────────┬─────────────────────┬───────────────┬──────────────────────┐
│ subject_id ┆ end_event_index ┆ prediction_time     ┆ boolean_value ┆ window_last_observed │
│ ---        ┆ ---             ┆ ---                 ┆ ---           ┆ ---                  │
│ i64        ┆ u32             ┆ datetime[μs]        ┆ bool          ┆ datetime[μs]         │
╞════════════╪═════════════════╪═════════════════════╪═══════════════╪══════════════════════╡
│ 239684     ┆ 3               ┆ 2010-05-11 18:00:00 ┆ false         ┆ 2010-05-11 17:48:48  │
│ 239684     ┆ 4               ┆ 2010-05-11 18:30:00 ┆ true          ┆ 2010-05-11 18:25:35  │
│ 239684     ┆ 5               ┆ 2010-05-11 19:00:00 ┆ true          ┆ 2010-05-11 18:57:18  │
│ 1195293    ┆ 3               ┆ 2010-06-20 19:30:00 ┆ false         ┆ 2010-06-20 19:25:32  │
│ 1195293    ┆ 4               ┆ 2010-06-20 20:00:00 ┆ true          ┆ 2010-06-20 19:45:19  │
│ …          ┆ …               ┆ …                   ┆ …             ┆ …                    │
│ 68729      ┆ 2               ┆ 2010-05-26 04:00:00 ┆ true          ┆ 2010-05-26 02:30:56  │
│ 68729      ┆ 2               ┆ 2010-05-26 04:30:00 ┆ true          ┆ 2010-05-26 02:30:56  │
│ 814703     ┆ 2               ┆ 2010-02-05 06:00:00 ┆ false         ┆ 2010-02-05 05:55:39  │
│ 814703     ┆ 2               ┆ 2010-02-05 06:30:00 ┆ true          ┆ 2010-02-05 05:55:39  │
│ 814703     ┆ 2               ┆ 2010-02-05 07:00:00 ┆ true          ┆ 2010-02-05 05:55:39  │
└────────────┴─────────────────┴─────────────────────┴───────────────┴──────────────────────┘

```

##### Returned items

While the raw data has codes as strings, naturally, when embedded in the pytorch dataset, they'll get
converted to integers. This happens during the forementioned tensorization step. We can see how the codes are
mapped to integers by looking at the output code metadata of that step:

```python
>>> code_metadata = pl.read_parquet(tensorized_MEDS_dataset.joinpath("metadata/codes.parquet"))
>>> code_metadata.select("code", "code/vocab_index")
shape: (11, 2)
┌───────────────────────┬──────────────────┐
│ code                  ┆ code/vocab_index │
│ ---                   ┆ ---              │
│ str                   ┆ u8               │
╞═══════════════════════╪══════════════════╡
│ ADMISSION//CARDIAC    ┆ 1                │
│ ADMISSION//ORTHOPEDIC ┆ 2                │
│ ADMISSION//PULMONARY  ┆ 3                │
│ DISCHARGE             ┆ 4                │
│ DOB                   ┆ 5                │
│ …                     ┆ …                │
│ EYE_COLOR//BROWN      ┆ 7                │
│ EYE_COLOR//HAZEL      ┆ 8                │
│ HEIGHT                ┆ 9                │
│ HR                    ┆ 10               │
│ TEMP                  ┆ 11               │
└───────────────────────┴──────────────────┘

```

We can see these vocab indices being used if we look at some elements of the pytorch dataset. Note that some
elements of the returned dictionaries are
[`JointNestedRaggedTensorDict`](https://github.com/mmcdermott/nested_ragged_tensors) objects, so we'll define
a helper here that will use a helper from the associated library to help us pretty-print out outputs. Note
that we'll also reduce precision in the numeric values to make the output more readable.

```python
>>> from nested_ragged_tensors.ragged_numpy import pprint_dense
>>> def print_element(el: dict):
...     for k, v in el.items():
...         print(f"{k} ({type(v).__name__}):")
...         if k == "dynamic":
...             pprint_dense(v.to_dense())
...         else:
...             print(v)
>>> print_element(pyd[2])
static_code (list):
[8, 9]
static_numeric_value (list):
[nan, -0.5438239574432373]
dynamic (JointNestedRaggedTensorDict):
code
[ 5  3 10 11  4]
.
numeric_value
[        nan         nan -1.4474752  -0.34049404         nan]
.
time_delta_days
[           nan 1.17661045e+04 0.00000000e+00 0.00000000e+00
 9.78703722e-02]

```

This example shows what the output looks like if we set the static data inclusion mode to `"include"`. What if
we set it to `"prepend"` instead? To show this in a stable manner, we'll also use the seeded version of the
get item function, `_seeded_getitem`:

```python
>>> pyd.config.static_inclusion_mode = "prepend"
>>> print_element(pyd._seeded_getitem(2, seed=0))
n_static_seq_els (int):
2
dynamic (JointNestedRaggedTensorDict):
code
[ 8  9  3 10 11]
.
numeric_value
[        nan -0.54382396         nan -1.4474752  -0.34049404]
.
time_delta_days
[       nan        nan 11766.1045     0.         0.    ]
>>> pyd.config.static_inclusion_mode = "include"

```

````

We can also look at what would be returned if we had included a task in the dataset:

```python
>>> print_element(pyd_with_task[0])
static_code (list):
[7, 9]
static_numeric_value (list):
[nan, 1.5770268440246582]
dynamic (JointNestedRaggedTensorDict):
code
[ 1 10 11 10 11]
.
numeric_value
[       nan -0.5697369 -1.2714673 -0.4375474 -1.1680276]
.
time_delta_days
[1.0726737e+04 0.0000000e+00 0.0000000e+00 4.8263888e-03 0.0000000e+00]
boolean_value (bool):
False

````

We can see in this case that the `boolean_value` field is included in the output, capturing the task label.

The contents of `pyd[2]` are stable, because index element 0, `(68729, 0, 3)`, indicates the first subject has
a sequence of length 3 in the dataset and our `max_seq_len` is set to 5.

```python
>>> print_element(pyd[2])
static_code (list):
[8, 9]
static_numeric_value (list):
[nan, -0.5438239574432373]
dynamic (JointNestedRaggedTensorDict):
code
[ 5  3 10 11  4]
.
numeric_value
[        nan         nan -1.4474752  -0.34049404         nan]
.
time_delta_days
[           nan 1.17661045e+04 0.00000000e+00 0.00000000e+00
 9.78703722e-02]

```

If we sampled a different subject, one with more than 5 events, the output we'd get would be dependent on the
`config.seq_sampling_strategy` option, and could be non-deterministic. By default, this is set to `random`, so
we'll get a random subset of length 5 each time. Here, so that this code is deterministic, we'll use
`_seeded_getitem`, an internal, seeded version of the `__getitem__` call.

```python
>>> print_element(pyd._seeded_getitem(1, seed=0))
static_code (list):
[6, 9]
static_numeric_value (list):
[nan, 0.06802856922149658]
dynamic (JointNestedRaggedTensorDict):
code
[10 11 10 11 10]
.
numeric_value
[-0.04626633  0.69391906 -0.30007038  0.79735875 -0.31064537]
.
time_delta_days
[0.01888889 0.         0.0084838  0.         0.01167824]
>>> print_element(pyd._seeded_getitem(1, seed=1))
static_code (list):
[6, 9]
static_numeric_value (list):
[nan, 0.06802856922149658]
dynamic (JointNestedRaggedTensorDict):
code
[10 11 10 11 10]
.
numeric_value
[ 0.03833488  0.79735875  0.33972722  0.7456389  -0.04626633]
.
time_delta_days
[0.00115741 0.         0.01373843 0.         0.01888889]

```

Of course, if we set `seq_sampling_strategy` to something other than `"random"`, this non-determinism would
disappear:

```python
>>> cfg_from_start = MEDSTorchDataConfig(
...     tensorized_cohort_dir=tensorized_MEDS_dataset, max_seq_len=5, seq_sampling_strategy="from_start"
... )
>>> pyd_from_start = MEDSPytorchDataset(cfg_from_start, split="train")
>>> print_element(pyd_from_start[1])
static_code (list):
[6, 9]
static_numeric_value (list):
[nan, 0.06802856922149658]
dynamic (JointNestedRaggedTensorDict):
code
[ 5  1 10 11 10]
.
numeric_value
[        nan         nan -0.23133166  0.79735875  0.03833488]
.
time_delta_days
[          nan 1.1688809e+04 0.0000000e+00 0.0000000e+00 1.1574074e-03]
>>> print_element(pyd_from_start[1])
static_code (list):
[6, 9]
static_numeric_value (list):
[nan, 0.06802856922149658]
dynamic (JointNestedRaggedTensorDict):
code
[ 5  1 10 11 10]
.
numeric_value
[        nan         nan -0.23133166  0.79735875  0.03833488]
.
time_delta_days
[          nan 1.1688809e+04 0.0000000e+00 0.0000000e+00 1.1574074e-03]


```

##### Batches, Collation, and Dataloaders

We can also examine not just individual elements, but full batches, that we can access with the appropriate
`collate` function via the built in `get_dataloader` method. Here, we'll treat these outputs like
dictionaries, but they actually return dataclass objects that have some additional properties we can use to
access shapes and validate data. See the
[API documentation](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/types#meds_torchdata.types.MEDSTorchBatch)
on the batch class for more information.

```python
>>> batches = [batch for batch in pyd.get_dataloader(batch_size=2)]
>>> print(batches[1])
MEDSTorchBatch:
│ Mode: Subject-Measurement (SM)
│ Static data? ✓
│ Labels? ✗
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length: 5
│ │
│ │ All dynamic data: (2, 5)
│ │ Static data: (2, 2)
│
│ Data:
│ │ Dynamic:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[0.00e+00, 1.18e+04,  ..., 0.00e+00, 9.79e-02],
│ │ │ │  [0.00e+00, 1.24e+04,  ..., 0.00e+00, 4.64e-02]]
│ │ │ code (torch.int64):
│ │ │ │ [[ 5,  3,  ..., 11,  4],
│ │ │ │  [ 5,  2,  ..., 11,  4]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[ 0.00,  0.00,  ..., -0.34,  0.00],
│ │ │ │  [ 0.00,  0.00,  ...,  0.85,  0.00]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[False, False,  ...,  True, False],
│ │ │ │  [False, False,  ...,  True, False]]
│ │
│ │ Static:
│ │ │ static_code (torch.int64):
│ │ │ │ [[8, 9],
│ │ │ │  [8, 9]]
│ │ │ static_numeric_value (torch.float32):
│ │ │ │ [[ 0.00, -0.54],
│ │ │ │  [ 0.00, -1.10]]
│ │ │ static_numeric_value_mask (torch.bool):
│ │ │ │ [[False,  True],
│ │ │ │  [False,  True]]
>>> print(next(iter(pyd_with_task.get_dataloader(batch_size=2))))
MEDSTorchBatch:
│ Mode: Subject-Measurement (SM)
│ Static data? ✓
│ Labels? ✓
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length: 5
│ │
│ │ All dynamic data: (2, 5)
│ │ Static data: (2, 2)
│ │ Labels: torch.Size([2])
│
│ Data:
│ │ Dynamic:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[1.07e+04, 0.00e+00,  ..., 4.83e-03, 0.00e+00],
│ │ │ │  [0.00e+00, 4.83e-03,  ..., 2.55e-02, 0.00e+00]]
│ │ │ code (torch.int64):
│ │ │ │ [[ 1, 10,  ..., 10, 11],
│ │ │ │  [11, 10,  ..., 10, 11]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[ 0.00e+00, -5.70e-01,  ..., -4.38e-01, -1.17e+00],
│ │ │ │  [-1.27e+00, -4.38e-01,  ...,  1.32e-03, -1.37e+00]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[False,  True,  ...,  True,  True],
│ │ │ │  [ True,  True,  ...,  True,  True]]
│ │
│ │ Static:
│ │ │ static_code (torch.int64):
│ │ │ │ [[7, 9],
│ │ │ │  [7, 9]]
│ │ │ static_numeric_value (torch.float32):
│ │ │ │ [[0.00, 1.58],
│ │ │ │  [0.00, 1.58]]
│ │ │ static_numeric_value_mask (torch.bool):
│ │ │ │ [[False,  True],
│ │ │ │  [False,  True]]
│ │
│ │ Labels:
│ │ │ boolean_value (torch.bool):
│ │ │ │ [False,  True]

```

This is with the default static inclusion mode of `"include"`, which means that the static data is included as
a separate entry in the batch. What about with the other two options, `"omit"` and `"prepend"`?

If we use `"omit"`, we can see that the static data is omitted from the output:

```python
>>> pyd_with_task.config.static_inclusion_mode = "omit"
>>> print(next(iter(pyd_with_task.get_dataloader(batch_size=2))))
MEDSTorchBatch:
│ Mode: Subject-Measurement (SM)
│ Static data? ✗
│ Labels? ✓
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length: 5
│ │
│ │ All dynamic data: (2, 5)
│ │ Labels: torch.Size([2])
│
│ Data:
│ │ Dynamic:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[1.07e+04, 0.00e+00,  ..., 4.83e-03, 0.00e+00],
│ │ │ │  [0.00e+00, 4.83e-03,  ..., 2.55e-02, 0.00e+00]]
│ │ │ code (torch.int64):
│ │ │ │ [[ 1, 10,  ..., 10, 11],
│ │ │ │  [11, 10,  ..., 10, 11]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[ 0.00e+00, -5.70e-01,  ..., -4.38e-01, -1.17e+00],
│ │ │ │  [-1.27e+00, -4.38e-01,  ...,  1.32e-03, -1.37e+00]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[False,  True,  ...,  True,  True],
│ │ │ │  [ True,  True,  ...,  True,  True]]
│ │
│ │ Labels:
│ │ │ boolean_value (torch.bool):
│ │ │ │ [False,  True]

```

What if we use a static inclusion mode of `"prepend"`? We can see that the static data is prepended to the
dynamic data:

```python
>>> pyd_with_task.config.static_inclusion_mode = "prepend"
>>> print(next(iter(pyd_with_task.get_dataloader(batch_size=2))))
MEDSTorchBatch:
│ Mode: Subject-Measurement (SM)
│ Static data? ✓ (prepended)
│ Labels? ✓
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length (static + dynamic): 5
│ │
│ │ All [static; dynamic] data: (2, 5)
│ │ Labels: torch.Size([2])
│
│ Data:
│ │ [Static; Dynamic]:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[0.00, 0.00,  ..., 0.00, 0.00],
│ │ │ │  [0.00, 0.00,  ..., 0.03, 0.00]]
│ │ │ code (torch.int64):
│ │ │ │ [[ 7,  9,  ..., 10, 11],
│ │ │ │  [ 7,  9,  ..., 10, 11]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[ 0.00e+00,  1.58e+00,  ..., -4.38e-01, -1.17e+00],
│ │ │ │  [ 0.00e+00,  1.58e+00,  ...,  1.32e-03, -1.37e+00]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[False,  True,  ...,  True,  True],
│ │ │ │  [False,  True,  ...,  True,  True]]
│ │ │ static_mask (torch.bool):
│ │ │ │ [[ True,  True,  ..., False, False],
│ │ │ │  [ True,  True,  ..., False, False]]
│ │
│ │ Labels:
│ │ │ boolean_value (torch.bool):
│ │ │ │ [False,  True]
>>> pyd_with_task.config.static_inclusion_mode = "include" # reset to default

```

Thus far, our examples have all worked with the default config object, which sets (among other things) the
default output to be at a _measurement_ level, rather than an _event_ level, by virtue of setting
`batch_mode` to `SM`. Let's see what happens if we change that:

```python
>>> pyd.config.batch_mode = "SEM"
>>> print_element(pyd[2])
static_code (list):
[8, 9]
static_numeric_value (list):
[nan, -0.5438239574432373]
dynamic (JointNestedRaggedTensorDict):
time_delta_days
[           nan 1.17661045e+04 9.78703722e-02]
.
---
.
dim1/mask
[[ True False False]
 [ True  True  True]
 [ True False False]]
.
code
[[ 5  0  0]
 [ 3 10 11]
 [ 4  0  0]]
.
numeric_value
[[        nan  0.          0.        ]
 [        nan -1.4474752  -0.34049404]
 [        nan  0.          0.        ]]
>>> batches = [batch for batch in pyd.get_dataloader(batch_size=2)]
>>> print(batches[1])
MEDSTorchBatch:
│ Mode: Subject-Event-Measurement (SEM)
│ Static data? ✓
│ Labels? ✗
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length: 3
│ │ Event length: 3
│ │
│ │ Per-event data: (2, 3)
│ │ Per-measurement data: (2, 3, 3)
│ │ Static data: (2, 2)
│
│ Data:
│ │ Event-level:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[0.00e+00, 1.18e+04, 9.79e-02],
│ │ │ │  [0.00e+00, 1.24e+04, 4.64e-02]]
│ │ │ event_mask (torch.bool):
│ │ │ │ [[True, True, True],
│ │ │ │  [True, True, True]]
│ │
│ │ Measurement-level:
│ │ │ code (torch.int64):
│ │ │ │ [[[ 5,  0,  0],
│ │ │ │   [ 3, 10, 11],
│ │ │ │   [ 4,  0,  0]],
│ │ │ │  [[ 5,  0,  0],
│ │ │ │   [ 2, 10, 11],
│ │ │ │   [ 4,  0,  0]]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[[ 0.00,  0.00,  0.00],
│ │ │ │   [ 0.00, -1.45, -0.34],
│ │ │ │   [ 0.00,  0.00,  0.00]],
│ │ │ │  [[ 0.00,  0.00,  0.00],
│ │ │ │   [ 0.00,  3.00,  0.85],
│ │ │ │   [ 0.00,  0.00,  0.00]]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   [False,  True,  True]],
│ │ │ │  [[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   [False,  True,  True]]]
│ │
│ │ Static:
│ │ │ static_code (torch.int64):
│ │ │ │ [[8, 9],
│ │ │ │  [8, 9]]
│ │ │ static_numeric_value (torch.float32):
│ │ │ │ [[ 0.00, -0.54],
│ │ │ │  [ 0.00, -1.10]]
│ │ │ static_numeric_value_mask (torch.bool):
│ │ │ │ [[False,  True],
│ │ │ │  [False,  True]]
>>> pyd_with_task.config.batch_mode = "SEM"
>>> print(next(iter(pyd_with_task.get_dataloader(batch_size=2))))
MEDSTorchBatch:
│ Mode: Subject-Event-Measurement (SEM)
│ Static data? ✓
│ Labels? ✓
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length: 4
│ │ Event length: 3
│ │
│ │ Per-event data: (2, 4)
│ │ Per-measurement data: (2, 4, 3)
│ │ Static data: (2, 2)
│ │ Labels: torch.Size([2])
│
│ Data:
│ │ Event-level:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[0.00e+00, 1.07e+04, 4.83e-03, 0.00e+00],
│ │ │ │  [0.00e+00, 1.07e+04, 4.83e-03, 2.55e-02]]
│ │ │ event_mask (torch.bool):
│ │ │ │ [[ True,  True,  True, False],
│ │ │ │  [ True,  True,  True,  True]]
│ │
│ │ Measurement-level:
│ │ │ code (torch.int64):
│ │ │ │ [[[ 5,  0,  0],
│ │ │ │   [ 1, 10, 11],
│ │ │ │   [10, 11,  0],
│ │ │ │   [ 0,  0,  0]],
│ │ │ │  [[ 5,  0,  0],
│ │ │ │   [ 1, 10, 11],
│ │ │ │   [10, 11,  0],
│ │ │ │   [10, 11,  0]]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[[ 0.00e+00,  0.00e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00, -5.70e-01, -1.27e+00],
│ │ │ │   [-4.38e-01, -1.17e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00,  0.00e+00,  0.00e+00]],
│ │ │ │  [[ 0.00e+00,  0.00e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00, -5.70e-01, -1.27e+00],
│ │ │ │   [-4.38e-01, -1.17e+00,  0.00e+00],
│ │ │ │   [ 1.32e-03, -1.37e+00,  0.00e+00]]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   [ True,  True,  True],
│ │ │ │   [ True,  True,  True]],
│ │ │ │  [[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   [ True,  True,  True],
│ │ │ │   [ True,  True,  True]]]
│ │
│ │ Static:
│ │ │ static_code (torch.int64):
│ │ │ │ [[7, 9],
│ │ │ │  [7, 9]]
│ │ │ static_numeric_value (torch.float32):
│ │ │ │ [[0.00, 1.58],
│ │ │ │  [0.00, 1.58]]
│ │ │ static_numeric_value_mask (torch.bool):
│ │ │ │ [[False,  True],
│ │ │ │  [False,  True]]
│ │
│ │ Labels:
│ │ │ boolean_value (torch.bool):
│ │ │ │ [False,  True]
>>> pyd_with_task.config.static_inclusion_mode = "omit"
>>> print(next(iter(pyd_with_task.get_dataloader(batch_size=2))))
MEDSTorchBatch:
│ Mode: Subject-Event-Measurement (SEM)
│ Static data? ✗
│ Labels? ✓
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length: 4
│ │ Event length: 3
│ │
│ │ Per-event data: (2, 4)
│ │ Per-measurement data: (2, 4, 3)
│ │ Labels: torch.Size([2])
│
│ Data:
│ │ Event-level:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[0.00e+00, 1.07e+04, 4.83e-03, 0.00e+00],
│ │ │ │  [0.00e+00, 1.07e+04, 4.83e-03, 2.55e-02]]
│ │ │ event_mask (torch.bool):
│ │ │ │ [[ True,  True,  True, False],
│ │ │ │  [ True,  True,  True,  True]]
│ │
│ │ Measurement-level:
│ │ │ code (torch.int64):
│ │ │ │ [[[ 5,  0,  0],
│ │ │ │   [ 1, 10, 11],
│ │ │ │   [10, 11,  0],
│ │ │ │   [ 0,  0,  0]],
│ │ │ │  [[ 5,  0,  0],
│ │ │ │   [ 1, 10, 11],
│ │ │ │   [10, 11,  0],
│ │ │ │   [10, 11,  0]]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[[ 0.00e+00,  0.00e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00, -5.70e-01, -1.27e+00],
│ │ │ │   [-4.38e-01, -1.17e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00,  0.00e+00,  0.00e+00]],
│ │ │ │  [[ 0.00e+00,  0.00e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00, -5.70e-01, -1.27e+00],
│ │ │ │   [-4.38e-01, -1.17e+00,  0.00e+00],
│ │ │ │   [ 1.32e-03, -1.37e+00,  0.00e+00]]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   [ True,  True,  True],
│ │ │ │   [ True,  True,  True]],
│ │ │ │  [[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   [ True,  True,  True],
│ │ │ │   [ True,  True,  True]]]
│ │
│ │ Labels:
│ │ │ boolean_value (torch.bool):
│ │ │ │ [False,  True]
>>> pyd_with_task.config.static_inclusion_mode = "prepend"
>>> print(next(iter(pyd_with_task.get_dataloader(batch_size=2))))
MEDSTorchBatch:
│ Mode: Subject-Event-Measurement (SEM)
│ Static data? ✓ (prepended)
│ Labels? ✓
│
│ Shape:
│ │ Batch size: 2
│ │ Sequence length (static + dynamic): 5
│ │ Event length: 3
│ │
│ │ Per-event data: (2, 5)
│ │ Per-measurement data: (2, 5, 3)
│ │ Labels: torch.Size([2])
│
│ Data:
│ │ Event-level:
│ │ │ time_delta_days (torch.float32):
│ │ │ │ [[0.00, 0.00,  ..., 0.00, 0.00],
│ │ │ │  [0.00, 0.00,  ..., 0.00, 0.03]]
│ │ │ event_mask (torch.bool):
│ │ │ │ [[ True,  True,  ...,  True, False],
│ │ │ │  [ True,  True,  ...,  True,  True]]
│ │ │ static_mask (torch.bool):
│ │ │ │ [[ True, False,  ..., False, False],
│ │ │ │  [ True, False,  ..., False, False]]
│ │
│ │ Measurement-level:
│ │ │ code (torch.int64):
│ │ │ │ [[[ 7,  9,  0],
│ │ │ │   [ 5,  0,  0],
│ │ │ │   ...,
│ │ │ │   [10, 11,  0],
│ │ │ │   [ 0,  0,  0]],
│ │ │ │  [[ 7,  9,  0],
│ │ │ │   [ 5,  0,  0],
│ │ │ │   ...,
│ │ │ │   [10, 11,  0],
│ │ │ │   [10, 11,  0]]]
│ │ │ numeric_value (torch.float32):
│ │ │ │ [[[ 0.00e+00,  1.58e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00,  0.00e+00,  0.00e+00],
│ │ │ │   ...,
│ │ │ │   [-4.38e-01, -1.17e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00,  0.00e+00,  0.00e+00]],
│ │ │ │  [[ 0.00e+00,  1.58e+00,  0.00e+00],
│ │ │ │   [ 0.00e+00,  0.00e+00,  0.00e+00],
│ │ │ │   ...,
│ │ │ │   [-4.38e-01, -1.17e+00,  0.00e+00],
│ │ │ │   [ 1.32e-03, -1.37e+00,  0.00e+00]]]
│ │ │ numeric_value_mask (torch.bool):
│ │ │ │ [[[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   ...,
│ │ │ │   [ True,  True,  True],
│ │ │ │   [ True,  True,  True]],
│ │ │ │  [[False,  True,  True],
│ │ │ │   [False,  True,  True],
│ │ │ │   ...,
│ │ │ │   [ True,  True,  True],
│ │ │ │   [ True,  True,  True]]]
│ │
│ │ Labels:
│ │ │ boolean_value (torch.bool):
│ │ │ │ [False,  True]

```

#### Data Tensorization and Pre-processing Details

_Full documentation for the preprocessing pipeline can be found
[here](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/preprocessing/)_

The `MTD_preprocess` command leverages [`hydra`](https://hydra.cc/) to manage the configuration and running
via the command line. You can see the available options by running the command with the `--help` flag:

```bash
== MTD_preprocess ==

MTD_preprocess is a command line tool for pre-processing MEDS data for use with meds_torchdata.

== Config ==

This is the config generated for this run:

MEDS_dataset_dir: ???
output_dir: ???
stage_runner_fp: null
do_overwrite: false
do_reshard: false
log_dir: ${output_dir}/.logs

You can override everything using the hydra `key=value` syntax; for example:

MTD_preprocess MEDS_dataset_dir=/path/to/dataset output_dir=/path/to/output do_overwrite=True
```

The `MTD_preprocess` command runs the following pre-processing stages:

1. _`fit_normalization`_: Fitting necessary parameters for normalization from the raw data (e.g., the mean and
    standard deviation of the `numeric_value` field).
2. _`fit_vocabulary_indices`_: Assigning unique vocabulary indices to each unique `code` in the data so that
    they can be transformed to numerical indices for tensorization.
3. _`normalization`_: Normalizing the data using the parameters fit in the `fit_normalization` stage to have a
    mean of 0 and a standard deviation of 1.
4. _`tokenization`_: Producing the schema files necessary for the tensorization stage.
5. _`tensorization`_: Producing the nested ragged tensor views of the data.

> [!NOTE]
> If you would like additional normalization options to be supported, please comment on the upstream issue in
> [MEDS-Transforms](https://github.com/mmcdermott/MEDS_transforms/issues/177), _and_ file an issue here to
> capture supporting additional options cleanly going forward.

> [!NOTE]
> You should perform any additional, _model specific pre-processing_ on the data _prior to running the
> `MTD_preprocess` command_ for your specific use-case. Indeed, if you wish to perform additional
> pre-processing, such as
>
> - Dropping numeric values entirely and converting to quantile-modified codes.
> - Drop infrequent codes or aggregate codes into higher-order categories.
> - Restrict subjects to a specific time-window
> - Drop subjects with infrequent values
> - Occlude outlier numeric values
> - etc.
>     You should perform these steps on the raw MEDS data _prior to running the tensorization command_. This
>     ensures that the data is modified as you desire in an efficient, transparent way and that the tensorization
>     step works with data in its final format to avoid any issues with discrepancies in code vocabulary, etc.

### Advanced features

You can also use this package natively with Hydra in modeling applications by adding the
`meds_torchdata.MEDSTorchDataConfig` to the Hydra config store. This will allow you to use it as though it
were a fully defined `.yaml` configuration file in your application configuration. To do this, you simply need
to run `MEDSTorchDataConfig.add_to_config_store()` in your application, specifying the group name in which you
plan to use the config in your application.

E.g., if you have a config file like this:

```yaml
dataset:
  _target_: meds_torchdata.MEDSPytorchDataset
  config: MEDSTorchDataConfig
```

Then in your main application, prior to `@hydra.main`, you can run:

```python
from meds_torchdata.config import MEDSTorchDataConfig

MEDSTorchDataConfig.add_to_config_store("dataset/config")
```

This will add the `MEDSTorchDataConfig` to the Hydra config store in the nested `dataset/config` group, which
will allow you to override its parameters from the command line and instantiate it into object form natively.

### Testing Models that Use this Package

If you use this package to build your model, we also expose some pytest fixtures that can be used to test your
models. These fixtures are designed to be used with the `pytest` testing framework. These fixtures are similar
to the four fixtures we used above in the [Examples and Detailed Usage](#examples-and-detailed-usage) section.
Namely, they are:

- `tensorized_MEDS_dataset` fixture that points to a Path containing the tensorized and schema files for
    the `simple_static_MEDS` dataset.
- `tensorized_MEDS_dataset_with_task` fixture that points to a tuple containing:
    - A Path containing the tensorized and schema files for the `simple_static_MEDS_dataset_with_task` dataset
    - A Path pointing to the root task directory for the dataset
    - The specific task name for the dataset. Task label files will be stored in a subdir of the root task
        directory with this name.
- `sample_pytorch_dataset`: This will yield a `MEDSPytorchDataset` object built using the
    `tensorized_MEDS_dataset` data, without a downstream task.
- `sample_pytorch_dataset_with_task`: This will yield a `MEDSPytorchDataset` object built using the
    `tensorized_MEDS_dataset_with_task` data, with the associated downstream task.

You can rely on these fixtures to test your model in the normal way, directly having your model train using
input batches derived from the sample datasets.

## Performance

See https://mmcdermott.github.io/meds-torch-data/dev/bench/ for performance benchmarks for all commits in this
repository. See [here](benchmark/run.py) for the benchmarking script. Note that these benchmarks are likely to
change over time so should be judged relative to the content of the associated commits, not in absolute terms
(e.g., we are likely to benchmark on more or more complex synthetic data, etc.).
