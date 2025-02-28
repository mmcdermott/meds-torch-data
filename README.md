# MEDS TorchData: A PyTorch Dataset Class for MEDS Datasets

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white"></a>
<a href="https://pypi.org/project/meds-torch-data/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.0.1a1-blue?logoColor=blue"></a>
<a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
<a href="https://codecov.io/github/mmcdermott/meds-torch-data"><img src="https://codecov.io/github/mmcdermott/meds-torch-data/graph/badge.svg?token=BV119L5JQJ"/></a>
<a href="https://github.com/mmcdermott/meds-torch-data/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/mmcdermott/meds-torch-data/actions/workflows/tests.yaml/badge.svg"></a>
<a href="https://github.com/mmcdermott/meds-torch-data/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/mmcdermott/meds-torch-data/actions/workflows/code-quality-main.yaml/badge.svg"></a>
<a href="https://github.com/mmcdermott/meds-torch-data/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/oufattole/meds-torch.svg"></a>
<a href="https://github.com/mmcdermott/meds-torch-data/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
<a href="https://github.com/mmcdermott/meds-torch-data#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>

## 🚀 Quick Start

### Step 1: Install

```bash
pip install meds-torch-data
```

### Step 2: Data Tensorization:

:::warning
If your dataset is not sharded by split, you need to run a reshard to split stage first!
:::

```bash
MEDS_tensorize input_dir=... output_dir=...
```

### Step 3: Use the dataset:

In your code, simply:

```python
from meds_torchdata import MEDSPytorchDataset

pyd = MEDSPytorchDataset(...)
```

## 📚 Documentation

### Design Principles

A good PyTorch dataset class should:

- Be easy to use
- Have a minimal, constant resource footprint (memory, CPU, start-up time) during model training and
  inference, _regardless of the overall dataset size_.
- Perform as much work as possible in _static, re-usable dataset pre-processing_, rather than upon
  construction or in the __getitem__ method.
- Induce effectively negligible computational overhead in the __getitem__ method relative to model training.
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

### API and Usage

#### Data Tensorization and Pre-processing

The `MEDS_tensorize` command-line utility is used to convert the input MEDS data into a format that can be
loaded into the PyTorch dataset class contained in this package. This command performs a very simple series of
steps:

1. Normalize the data into an appropriate, numerical format, including:
   \- Assigning each unique `code` in the data a unique integer index and converting the codes to those
   integer indices.
   \- Normalizing the `numeric_value` field to have a mean of 0 and a standard deviation of 1. _If you would
   like additional normalization options supported, such as min-max normalization, please file a GitHub
   issue._
2. Produce a set of static, "schema" files that contain the unique time-points of each subjects' events as
   well as their static measurements.
3. Produce a set of `JointNestedRaggedTensorDict` object files that contain each subjects' dynamic
   measurements in the form of nested, ragged tensors that can be efficiently loaded via the associated
   [package](https://github.com/mmcdermott/nested_ragged_tensors)

These are the only three steps this pipeline performs. Note, however, that this does not mean you can't or
shouldn't perform additional, _model specific pre-processing_ on the data _prior to running the tensorization
command_ for your specific use-case. Indeed, if you wish to perform additional pre-processing, such as

- Dropping numeric values entirely and converting to quantile-modified codes.
- Drop infrequent codes or aggregate codes into higher-order categories.
- Restrict subjects to a specific time-window
- Drop subjects with infrequent values
- Occlude outlier numeric values
- etc.
  You should perform these steps on the raw MEDS data _prior to running the tensorization command_. This ensures
  that the data is modified as you desire in an efficient, transparent way and that the tensorization step works
  with data in its final format to avoid any issues with discrepancies in code vocabulary, etc.

#### Dataset Class

Once the data has been tensorized, you can use the `MEDSPytorchDataset` class to load the data into a PyTorch
dataset suitable to begin modeling! This dataset class takes a configuration object as input, with the
following fields:
