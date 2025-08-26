# MEDS TorchData Pre-processing: Tokenization and Tensorization

This directory contains the [MEDS-Transforms](https://meds-transforms.readthedocs.io/en/stable/)
transformations and overall CLI tool for executing the transformations necessary to pre-process MEDS data for
use with the `meds_torchdata` package. The code in this repository exposes the following command-line
utilities:

- [`MTD_tokenize`](tokenization.py): Produces the schema files necessary for production of the nested ragged
    tensor views and the static schema files. You should almost never run this command directly. It will be
    run as part of the `MTD_preprocess` command.
- [`MTD_tensorize`](tensorization.py): Produces the nested ragged tensor views of the data. You should
    almost never run this command directly. It will be run as part of the `MTD_preprocess` command.
- [`MTD_preprocess`](__main__.py): A wrapper around all of the necessary stages (both built in stages within
    [MEDS-Transforms](https://meds-transforms.readthedocs.io/en/stable/) and the `MTD_tokenize` and
    `MTD_tensorize` stages) to pre-process the data for use with the `meds_torchdata` package. _This command is
    what you should run to pre-process the data for use with the `meds_torchdata` package._

## `MTD_preprocess`

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

## Pre-process Stages

The pre-processing pipeline within `MTD_preprocess` can be fully understood by inspecting the
[`_MTD_preprocess.yaml`](configs/_MTD_preprocess.yaml) pipeline configuration file. This file is in the
MEDS-Transforms format, and you can see the list of stages that are run via the `stages` key:

```yaml
...
stages:
  - fit_normalization
  - fit_vocabulary_indices
  - normalization
  - tokenization
  - tensorization
```

These stages constitute the minimum necessary steps to leverage this package, and they include;

1. _`fit_normalization`_: Fitting necessary parameters for normalization from the raw data (e.g., the mean and
    standard deviation of the `numeric_value` field).
2. _`fit_vocabulary_indices`_: Assigning unique vocabulary indices to each unique `code` in the data so that
    they can be transformed to numerical indices for tensorization.
3. _`normalization`_: Normalizing the data using the parameters fit in the `fit_normalization` stage to have a
    mean of 0 and a standard deviation of 1.
4. _`tokenization`_: Producing the schema files necessary for the tensorization stage, leveraging the
    `MTD_tokenize` command.
5. _`tensorization`_: Producing the nested ragged tensor views of the data, leveraging the `MTD_tensorize`
    command.

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

## Re-sharding

> [!WARNING]
> If your dataset is not sharded by split, you need to run a reshard to split stage first! You can enable this
> by adding the `do_reshard=True` argument to the `MTD_preprocess` command.

## Controlling parallelism via the `stage_runner_fp`

You can parallelize the running of this pipeline by leveraging the built-in functionality of the
MEDS-Transforms library. Unfortunately, documentation on this feature is currently lacking; please file a
GitHub issue if you intend to use this feature to help prioritize adding appropriate documentation therein.
