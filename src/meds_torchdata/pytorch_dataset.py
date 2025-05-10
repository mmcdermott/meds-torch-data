import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .config import MEDSTorchDataConfig, StaticInclusionMode
from .types import BatchMode, MEDSTorchBatch, StaticData

logger = logging.getLogger(__name__)


class MEDSPytorchDataset(torch.utils.data.Dataset):
    """A PyTorch dataset that provides efficient PyTorch access to a MEDS dataset.

    This dataset is designed to work with data from the MEDS (Medical Event Data Set) format, supporting
    various types of medical events, static patient information, and task-specific labels. It provides
    functionality for loading, processing, and collating data for use in PyTorch models in an efficient manner
    that takes advantage of the sparsity of EHR data to minimize memory usage and computational time.

    Key design principles:
      1. The class will store an `index` variable that specifies what is the valid range of data to consider
         for any given subject in the dataset corresponding to an integer index passed to `__getitem__`.
      2. Data will only be loaded for subjects on an as-needed basis, and will not be cached, to minimize
         memory usage during normal operation.
      3. As much work as possible should be relegated to separate dataset pre-processing (resulting in files
         stored on disk) rather than this class to streamline operation.
      4. The primary input to this class in terms of data is a pre-processed set of "schema files" and "nested
         ragged tensor" data files that can be used to identify the shape of the dataset and to efficiently
         load the relevant tensor data, respectively.

    Args:
        cfg: Configuration options for the dataset, realized through a dataclass instance.
        split: The data split to use. This must match up to the splits stored in the root dataset's
               `metadata/subject_splits.parquet` file's `split` column.

    Attributes:
        config: The configuration options for the dataset.
        split: The data split to use.
        schema_dfs_by_shard: A dictionary mapping shard names to the schema DataFrames for that shard.
        subj_locations: A dictionary mapping subject IDs to their locations in the schema DataFrames.
        index: A list of tuples, where each tuple contains the subject ID and the end index for that subject.
        labels: The task labels for the dataset, if any. This will be `None` if there is no task.

    For examples of this class, see the global README.md. Here, we'll include some examples of other aspects
    of the class, such as error validation and specific methods.

    Examples:
        >>> cfg = MEDSTorchDataConfig(tensorized_cohort_dir=tensorized_MEDS_dataset, max_seq_len=5)
        >>> pyd = MEDSPytorchDataset(cfg, split="train")
        >>> len(pyd)
        4
        >>> pyd.index
        [(239684, 6), (1195293, 8), (68729, 3), (814703, 3)]

    If you pass in a non-existent split, you'll get an error as it won't be able to find the schema files:

        >>> pyd = MEDSPytorchDataset(cfg, split="nonexistent")
        Traceback (most recent call last):
            ...
        FileNotFoundError: No schema files found in /tmp/.../tokenization/schemas! If your data is not sharded
        by split, this error may occur because this codebase does not handle non-split sharded data. See Issue
        #79 for tracking this issue.
    """

    LABEL_COL = LabelSchema.boolean_value_name
    END_IDX = "end_event_index"

    @classmethod
    def get_task_seq_bounds_and_labels(cls, label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
        """Returns the event-level allowed input sequence boundaries and labels for each task sample.

        This function is guaranteed to output an index of the same order and length as `label_df`. Subjects
        not present in `schema_df` will be included in the output, with null labels and indices.

        Args:
            label_df: The DataFrame containing the task labels, in the MEDS Label DF schema.
            schema_df: A DataFrame with subject ID and a list of event timestamps for each shard.

        Returns:
            A copy of the labels DataFrame, restricted to included subjects, with the appropriate end indices
            for each task sample. Labels will be present if the `cls.LABEL_COL` is present in the input.

        Examples:
            >>> label_df = pl.DataFrame({
            ...     "subject_id": [1, 2, 2, 4, 3, 3, 3],
            ...     "prediction_time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 1, 1), datetime(2020, 1, 2),
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3),
            ...     ],
            ...     "boolean_value": [True, False, True, False, True, False, True],
            ... })
            >>> schema_df = pl.DataFrame({
            ...     "subject_id": [2, 6, 1, 3],
            ...     "time": [
            ...         # Subject 2: Prediction times are 2020-1-1,2020-1-2
            ...         [
            ...             datetime(2019, 12, 31),
            ...             datetime(2019, 12, 31, 12),
            ...             datetime(2019, 12, 31, 23, 59, 59),
            ...             datetime(2020, 1, 1, 0, 0, 1),
            ...             datetime(2020, 1, 2),
            ...             datetime(2020, 1, 20),
            ...         ],
            ...         # Subject 6: No prediction times
            ...         [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            ...         # Subject 1: Prediction times are 2020-1-1
            ...         [datetime(2019, 12, 1), datetime(2020, 1, 1), datetime(2020, 1, 2)],
            ...         # Subject 3: Prediction times are 2020-1-1,2020-1-2,2020-1-3
            ...         [datetime(2020, 1, 1), datetime(2021, 11, 2), datetime(2021, 11, 3)],
            ...     ],
            ... })
            >>> MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)
            shape: (6, 4)
            ┌────────────┬─────────────────┬─────────────────────┬───────────────┐
            │ subject_id ┆ end_event_index ┆ prediction_time     ┆ boolean_value │
            │ ---        ┆ ---             ┆ ---                 ┆ ---           │
            │ i64        ┆ u32             ┆ datetime[μs]        ┆ bool          │
            ╞════════════╪═════════════════╪═════════════════════╪═══════════════╡
            │ 1          ┆ 2               ┆ 2020-01-01 00:00:00 ┆ true          │
            │ 2          ┆ 3               ┆ 2020-01-01 00:00:00 ┆ false         │
            │ 2          ┆ 5               ┆ 2020-01-02 00:00:00 ┆ true          │
            │ 3          ┆ 1               ┆ 2020-01-01 00:00:00 ┆ true          │
            │ 3          ┆ 1               ┆ 2020-01-02 00:00:00 ┆ false         │
            │ 3          ┆ 1               ┆ 2020-01-03 00:00:00 ┆ true          │
            └────────────┴─────────────────┴─────────────────────┴───────────────┘
            >>> MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df.drop("boolean_value"), schema_df)
            shape: (6, 3)
            ┌────────────┬─────────────────┬─────────────────────┐
            │ subject_id ┆ end_event_index ┆ prediction_time     │
            │ ---        ┆ ---             ┆ ---                 │
            │ i64        ┆ u32             ┆ datetime[μs]        │
            ╞════════════╪═════════════════╪═════════════════════╡
            │ 1          ┆ 2               ┆ 2020-01-01 00:00:00 │
            │ 2          ┆ 3               ┆ 2020-01-01 00:00:00 │
            │ 2          ┆ 5               ┆ 2020-01-02 00:00:00 │
            │ 3          ┆ 1               ┆ 2020-01-01 00:00:00 │
            │ 3          ┆ 1               ┆ 2020-01-02 00:00:00 │
            │ 3          ┆ 1               ┆ 2020-01-03 00:00:00 │
            └────────────┴─────────────────┴─────────────────────┘
        """

        end_idx_expr = (
            pl.col(DataSchema.time_name)
            .search_sorted(pl.col(LabelSchema.prediction_time_name), side="right")
            .last()
            .alias(cls.END_IDX)
        )

        group_cols = ["_row", DataSchema.subject_id_name, LabelSchema.prediction_time_name]
        out_cols = [DataSchema.subject_id_name, cls.END_IDX, LabelSchema.prediction_time_name]

        if cls.LABEL_COL in label_df.collect_schema().names():
            group_cols.append(cls.LABEL_COL)
            out_cols.append(cls.LABEL_COL)

        return (
            label_df.join(schema_df, on=DataSchema.subject_id_name, how="inner", maintain_order="left")
            .with_row_index("_row")
            .explode(DataSchema.time_name)
            .group_by(group_cols, maintain_order=True)
            .agg(end_idx_expr)
            .select(out_cols)
        )

    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__()

        self.config: MEDSTorchDataConfig = cfg
        self.split: str = split

        logger.info("Reading subject schema and static data")

        self.schema_dfs_by_shard: dict[str, pl.DataFrame] = {}
        self.subj_locations: dict[int, tuple[str, int]] = {}

        for shard, schema_fp in self.config.schema_fps:
            if not shard.startswith(f"{self.split}/"):
                continue

            df = pl.read_parquet(schema_fp, use_pyarrow=True).with_columns(
                pl.col("static_code").list.eval(pl.element().fill_null(0)),
                pl.col("static_numeric_value").list.eval(pl.element().fill_null(np.nan)),
            )

            self.schema_dfs_by_shard[shard] = df
            for i, subj in enumerate(df[DataSchema.subject_id_name]):
                self.subj_locations[subj] = (shard, i)

        if not self.schema_dfs_by_shard:
            raise FileNotFoundError(
                f"No schema files found in {self.config.schema_dir}! If your data is not sharded by split, "
                "this error may occur because this codebase does not handle non-split sharded data. See "
                "Issue #79 for tracking this issue."
            )

        self.index = list(
            zip(self.schema_df[DataSchema.subject_id_name], self.schema_df[self.END_IDX], strict=False)
        )
        self.labels = self.schema_df[self.LABEL_COL] if self.has_task_labels else None

    @property
    def labels_df(self) -> pl.DataFrame:
        """Returns the task labels as a DataFrame, in the MEDS Label schema, or `None` if there is no task.

        Examples:
            >>> print(sample_pytorch_dataset.labels_df)
            None
            >>> sample_pytorch_dataset_with_task.labels_df
            shape: (21, 3)
            ┌────────────┬─────────────────────┬───────────────┐
            │ subject_id ┆ prediction_time     ┆ boolean_value │
            │ ---        ┆ ---                 ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ bool          │
            ╞════════════╪═════════════════════╪═══════════════╡
            │ 239684     ┆ 2010-05-11 18:00:00 ┆ false         │
            │ 239684     ┆ 2010-05-11 18:30:00 ┆ true          │
            │ 239684     ┆ 2010-05-11 19:00:00 ┆ true          │
            │ 1195293    ┆ 2010-06-20 19:30:00 ┆ false         │
            │ 1195293    ┆ 2010-06-20 20:00:00 ┆ true          │
            │ …          ┆ …                   ┆ …             │
            │ 754281     ┆ 2010-01-03 08:00:00 ┆ true          │
            │ 1500733    ┆ 2010-06-03 15:00:00 ┆ false         │
            │ 1500733    ┆ 2010-06-03 15:30:00 ┆ false         │
            │ 1500733    ┆ 2010-06-03 16:00:00 ┆ true          │
            │ 1500733    ┆ 2010-06-03 16:30:00 ┆ true          │
            └────────────┴─────────────────────┴───────────────┘
            >>> sample_pytorch_dataset_with_index.labels_df
            shape: (21, 2)
            ┌────────────┬─────────────────────┐
            │ subject_id ┆ prediction_time     │
            │ ---        ┆ ---                 │
            │ i64        ┆ datetime[μs]        │
            ╞════════════╪═════════════════════╡
            │ 239684     ┆ 2010-05-11 18:00:00 │
            │ 239684     ┆ 2010-05-11 18:30:00 │
            │ 239684     ┆ 2010-05-11 19:00:00 │
            │ 1195293    ┆ 2010-06-20 19:30:00 │
            │ 1195293    ┆ 2010-06-20 20:00:00 │
            │ …          ┆ …                   │
            │ 754281     ┆ 2010-01-03 08:00:00 │
            │ 1500733    ┆ 2010-06-03 15:00:00 │
            │ 1500733    ┆ 2010-06-03 15:30:00 │
            │ 1500733    ┆ 2010-06-03 16:00:00 │
            │ 1500733    ┆ 2010-06-03 16:30:00 │
            └────────────┴─────────────────────┘
        """
        if not self.has_task_index:
            return None

        required_cols = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        def read_df(fp: Path) -> pl.DataFrame:
            schema = pq.read_schema(fp)
            label_cols = [*required_cols, self.LABEL_COL] if self.LABEL_COL in schema.names else required_cols
            return pl.read_parquet(fp, columns=label_cols, use_pyarrow=True)

        logger.info(f"Reading tasks from {self.config.task_labels_fps}")
        return pl.concat([read_df(fp) for fp in self.config.task_labels_fps], how="vertical")

    @cached_property
    def schema_df(self) -> pl.DataFrame:
        """Returns the "schema" of this dataframe, cataloging each sample that will be output by row.

        This takes into account both task and non-task data, and is useful for aligning dataloader or model
        outputs to the source inputs.

        Examples:
            >>> sample_pytorch_dataset.schema_df
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
            >>> sample_pytorch_dataset_with_task.schema_df
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
            >>> sample_pytorch_dataset_with_index.schema_df
            shape: (13, 3)
            ┌────────────┬─────────────────┬─────────────────────┐
            │ subject_id ┆ end_event_index ┆ prediction_time     │
            │ ---        ┆ ---             ┆ ---                 │
            │ i64        ┆ u32             ┆ datetime[μs]        │
            ╞════════════╪═════════════════╪═════════════════════╡
            │ 239684     ┆ 3               ┆ 2010-05-11 18:00:00 │
            │ 239684     ┆ 4               ┆ 2010-05-11 18:30:00 │
            │ 239684     ┆ 5               ┆ 2010-05-11 19:00:00 │
            │ 1195293    ┆ 3               ┆ 2010-06-20 19:30:00 │
            │ 1195293    ┆ 4               ┆ 2010-06-20 20:00:00 │
            │ …          ┆ …               ┆ …                   │
            │ 68729      ┆ 2               ┆ 2010-05-26 04:00:00 │
            │ 68729      ┆ 2               ┆ 2010-05-26 04:30:00 │
            │ 814703     ┆ 2               ┆ 2010-02-05 06:00:00 │
            │ 814703     ┆ 2               ┆ 2010-02-05 06:30:00 │
            │ 814703     ┆ 2               ┆ 2010-02-05 07:00:00 │
            └────────────┴─────────────────┴─────────────────────┘
        """

        base_df = pl.concat(
            (
                df.select(DataSchema.subject_id_name, DataSchema.time_name)
                for df in self.schema_dfs_by_shard.values()
            ),
            how="vertical",
        )

        if self.has_task_index:
            return self.get_task_seq_bounds_and_labels(self.labels_df, base_df)
        else:
            return base_df.select(
                DataSchema.subject_id_name, pl.col(DataSchema.time_name).list.len().alias(self.END_IDX)
            )

    def __len__(self):
        """Returns the length of the dataset.

        Examples:
            >>> len(sample_pytorch_dataset)
            4
            >>> len(sample_pytorch_dataset_with_task)
            13
        """
        return len(self.index)

    @property
    def has_task_index(self) -> bool:
        """Returns whether the dataset has a task index specified.

        A convenience wrapper around the config property.

        Examples:
            >>> sample_pytorch_dataset.has_task_index
            False
            >>> sample_pytorch_dataset_with_index.has_task_index
            True
            >>> sample_pytorch_dataset_with_task.has_task_index
            True
        """
        return self.config.task_labels_dir is not None

    @property
    def has_task_labels(self) -> bool:
        """Returns whether the dataset has a task specified with labels.

        Examples:
            >>> sample_pytorch_dataset.has_task_labels
            False
            >>> sample_pytorch_dataset_with_index.has_task_labels
            False
            >>> sample_pytorch_dataset_with_task.has_task_labels
            True
        """
        return self.has_task_index and (self.LABEL_COL in self.schema_df.collect_schema().names())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset.

        This method returns a dictionary corresponding to a single subject's data at the specified index. The
        data is not tensorized in this method, as that work is typically done in the collate function.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            A dictionary containing the static code, static numeric value, dynamic data, and task label (if
            present) for the specified subject.
        """
        return self._seeded_getitem(idx)

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset with a specified random seed.

        This is merely a deterministic wrapper around the `_getitem` method that allows for deterministic
        subsequence sampling.
        """

        subject_id, end_idx = self.index[idx]
        dynamic_data, static_data = self.load_subject_data(subject_id=subject_id, st=0, end=end_idx)

        match self.config.static_inclusion_mode:
            case StaticInclusionMode.OMIT:
                out = {}
                n_static_measurements = None
            case StaticInclusionMode.INCLUDE:
                n_static_measurements = None
                out = {
                    "static_code": static_data.code,
                    "static_numeric_value": static_data.numeric_value,
                }
            case StaticInclusionMode.PREPEND:
                n_static_measurements = len(static_data.code)
                out = {"n_static_measurements": n_static_measurements}
            case _:
                raise NotImplementedError(
                    f"Static inclusion mode {self.config.static_inclusion_mode} not implemented."
                )

        dynamic_data = self.config.process_dynamic_data(
            dynamic_data, n_static_measurements=n_static_measurements, rng=seed
        )

        if self.config.static_inclusion_mode == StaticInclusionMode.PREPEND:
            if self.config.batch_mode == BatchMode.SEM:
                static_dict = {
                    "time_delta_days": [np.nan],
                    "code": [static_data.code],
                    "numeric_value": [static_data.numeric_value],
                }
            else:
                static_dict = {
                    "time_delta_days": [np.nan for _ in range(n_static_measurements)],
                    "code": static_data.code,
                    "numeric_value": static_data.numeric_value,
                }

            static_as_JNRT = JointNestedRaggedTensorDict(static_dict, schema=dynamic_data.schema)

            dynamic_data = JointNestedRaggedTensorDict.concatenate([static_as_JNRT, dynamic_data])

        out["dynamic"] = dynamic_data

        if self.has_task_labels:
            out[self.LABEL_COL] = self.labels[idx]

        return out

    def load_subject_data(
        self, subject_id: int, st: int, end: int
    ) -> tuple[JointNestedRaggedTensorDict, StaticData]:
        """Loads and returns the dynamic data slice for a given subject ID and permissible event range.

        Args:
            subject_id: The ID of the subject to load.
            st: The (integral) index of the first permissible event (meaning unique timestamp) that can be
                read for this subject's record. If None, no limit is applied.
            end: The (integral) index of the last permissible event (meaning unique timestamp) that can be
                 read for this subject's record. If None, no limit is applied.

        Returns:
            The subject's dynamic data and static data. The static data is returned as a StaticData named
            tuple with two fields: `code` and `numeric_value`.

        Examples:
            >>> from nested_ragged_tensors.ragged_numpy import pprint_dense
            >>> dynamic_data, static_data = sample_pytorch_dataset.load_subject_data(68729, 0, 3)
            >>> static_data.code
            [8, 9]
            >>> static_data.numeric_value
            [nan, -0.5438239574432373]
            >>> pprint_dense(dynamic_data.to_dense())
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

            To see that these make sense, recall we can check the raw data. Obviously, the data have been
            normalized and tokenized, so we should not expect exact matches in the numeric values or code
            strings, but were we to inspect the code vocabularies, they would align:

            >>> from meds_testing_helpers.dataset import MEDSDataset
            >>> D = MEDSDataset(root_dir=simple_static_MEDS)
            >>> raw_data = pl.from_arrow(D.data_shards["train/1"]).filter(pl.col("subject_id") == 68729)
            >>> raw_data
            shape: (7, 4)
            ┌────────────┬─────────────────────┬──────────────────────┬───────────────┐
            │ subject_id ┆ time                ┆ code                 ┆ numeric_value │
            │ ---        ┆ ---                 ┆ ---                  ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ str                  ┆ f32           │
            ╞════════════╪═════════════════════╪══════════════════════╪═══════════════╡
            │ 68729      ┆ null                ┆ EYE_COLOR//HAZEL     ┆ null          │
            │ 68729      ┆ null                ┆ HEIGHT               ┆ 160.395309    │
            │ 68729      ┆ 1978-03-09 00:00:00 ┆ DOB                  ┆ null          │
            │ 68729      ┆ 2010-05-26 02:30:56 ┆ ADMISSION//PULMONARY ┆ null          │
            │ 68729      ┆ 2010-05-26 02:30:56 ┆ HR                   ┆ 86.0          │
            │ 68729      ┆ 2010-05-26 02:30:56 ┆ TEMP                 ┆ 97.800003     │
            │ 68729      ┆ 2010-05-26 04:51:52 ┆ DISCHARGE            ┆ null          │
            └────────────┴─────────────────────┴──────────────────────┴───────────────┘
            >>> subj_codes = raw_data["code"].unique().to_list()
            >>> code_metadata = (
            ...     pl.read_parquet(tensorized_MEDS_dataset / "metadata/codes.parquet")
            ...     .filter(pl.col("code").is_in(subj_codes))
            ... )
            >>> mean_col = (pl.col("values/sum")/pl.col("values/n_occurrences")).alias("values/mean")
            >>> std_col = (
            ...     (pl.col("values/sum_sqd")/pl.col("values/n_occurrences") - mean_col**2)**0.5
            ... ).alias("values/std")
            >>> code_metadata.select("code", "code/vocab_index", mean_col, std_col)
            shape: (7, 4)
            ┌──────────────────────┬──────────────────┬─────────────┬────────────┐
            │ code                 ┆ code/vocab_index ┆ values/mean ┆ values/std │
            │ ---                  ┆ ---              ┆ ---         ┆ ---        │
            │ str                  ┆ u8               ┆ f32         ┆ f32        │
            ╞══════════════════════╪══════════════════╪═════════════╪════════════╡
            │ ADMISSION//PULMONARY ┆ 3                ┆ NaN         ┆ NaN        │
            │ DISCHARGE            ┆ 4                ┆ NaN         ┆ NaN        │
            │ DOB                  ┆ 5                ┆ NaN         ┆ NaN        │
            │ EYE_COLOR//HAZEL     ┆ 8                ┆ NaN         ┆ NaN        │
            │ HEIGHT               ┆ 9                ┆ 164.209732  ┆ 7.014076   │
            │ HR                   ┆ 10               ┆ 113.375     ┆ 18.912241  │
            │ TEMP                 ┆ 11               ┆ 98.458336   ┆ 1.933464   │
            └──────────────────────┴──────────────────┴─────────────┴────────────┘

            Note this is independent of the task data and the index; this only depends on the raw data on
            disk. So, we'll see the exact same output if we call over the sample dataset with tasks because
            the raw MEDS data is the same.

            >>> dynamic_data, static_data = sample_pytorch_dataset_with_task.load_subject_data(68729, 0, 3)
            >>> static_data.code
            [8, 9]
            >>> static_data.numeric_value
            [nan, -0.5438239574432373]
            >>> pprint_dense(dynamic_data.to_dense())
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
        """
        shard, subject_idx = self.subj_locations[subject_id]

        dynamic_data_fp = self.config.tensorized_cohort_dir / "data" / f"{shard}.nrt"
        subject_dynamic_data = JointNestedRaggedTensorDict(tensors_fp=dynamic_data_fp)[subject_idx, st:end]

        subj_schema = self.schema_dfs_by_shard[shard][subject_idx]
        static_code = subj_schema["static_code"].item().to_list()
        static_numeric_value = subj_schema["static_numeric_value"].item().to_list()

        return subject_dynamic_data, StaticData(static_code, static_numeric_value)

    def collate(self, batch: list[dict]) -> MEDSTorchBatch:
        """Combines a batch of data points into a single, tensorized batch.

        The collated output is a fully tensorized and padded dictionary, ready for input into an
        `input_encoder`. This method uses the JointNestedRaggedTensorDict API to collate and pad the data.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single sample as
                returned by the __getitem__ method.

        Returns:
            MEDSTorchBatch: A simple, dictionary-like object containing the collated batch data. See the
            [method documentation](../types.py) for more information.

        Examples:
            >>> raw_batch = [sample_pytorch_dataset[2], sample_pytorch_dataset[3]]
            >>> print(sample_pytorch_dataset.collate(raw_batch))
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
            >>> raw_batch = [sample_pytorch_dataset_with_task[0], sample_pytorch_dataset_with_task[1]]
            >>> print(sample_pytorch_dataset_with_task.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Measurement (SM)
            │ Static data? ✓
            │ Labels? ✓
            │
            │ Shape:
            │ │ Batch size: 2
            │ │ Sequence length: 8
            │ │
            │ │ All dynamic data: (2, 8)
            │ │ Static data: (2, 2)
            │ │ Labels: torch.Size([2])
            │
            │ Data:
            │ │ Dynamic:
            │ │ │ time_delta_days (torch.float32):
            │ │ │ │ [[0.00e+00, 1.07e+04,  ..., 0.00e+00, 0.00e+00],
            │ │ │ │  [0.00e+00, 1.07e+04,  ..., 2.55e-02, 0.00e+00]]
            │ │ │ code (torch.int64):
            │ │ │ │ [[ 5,  1,  ...,  0,  0],
            │ │ │ │  [ 5,  1,  ..., 10, 11]]
            │ │ │ numeric_value (torch.float32):
            │ │ │ │ [[ 0.00e+00,  0.00e+00,  ...,  0.00e+00,  0.00e+00],
            │ │ │ │  [ 0.00e+00,  0.00e+00,  ...,  1.32e-03, -1.37e+00]]
            │ │ │ numeric_value_mask (torch.bool):
            │ │ │ │ [[False, False,  ...,  True,  True],
            │ │ │ │  [False, False,  ...,  True,  True]]
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

            You can also change the padding side. This defaults to "right" (which is typical for modeling) but
            you can set it to "left" for generative use cases. To show this, we'll also set the sampling
            strategy to `SubsequenceSamplingStrategy.TO_END` so that things are consistent.

            >>> from meds_torchdata.types import SubsequenceSamplingStrategy
            >>> sample_pytorch_dataset.config.padding_side = "left"
            >>> sample_pytorch_dataset.config.seq_sampling_strategy = SubsequenceSamplingStrategy.TO_END
            >>> raw_batch = [sample_pytorch_dataset[i] for i in range(len(sample_pytorch_dataset))]
            >>> print(sample_pytorch_dataset.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Measurement (SM)
            │ Static data? ✓
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 4
            │ │ Sequence length: 10
            │ │
            │ │ All dynamic data: (4, 10)
            │ │ Static data: (4, 2)
            │
            │ Data:
            │ │ Dynamic:
            │ │ │ time_delta_days (torch.float32):
            │ │ │ │ [[1.07e+04, 0.00e+00,  ..., 0.00e+00, 2.08e-02],
            │ │ │ │  [0.00e+00, 1.37e-02,  ..., 0.00e+00, 5.91e-03],
            │ │ │ │  [0.00e+00, 0.00e+00,  ..., 0.00e+00, 9.79e-02],
            │ │ │ │  [0.00e+00, 0.00e+00,  ..., 0.00e+00, 4.64e-02]]
            │ │ │ code (torch.int64):
            │ │ │ │ [[ 1, 10,  ..., 11,  4],
            │ │ │ │  [11, 10,  ..., 11,  4],
            │ │ │ │  [ 0,  0,  ..., 11,  4],
            │ │ │ │  [ 0,  0,  ..., 11,  4]]
            │ │ │ numeric_value (torch.float32):
            │ │ │ │ [[ 0.00, -0.57,  ..., -1.53,  0.00],
            │ │ │ │  [ 0.80,  0.34,  ...,  1.00,  0.00],
            │ │ │ │  [ 0.00,  0.00,  ..., -0.34,  0.00],
            │ │ │ │  [ 0.00,  0.00,  ...,  0.85,  0.00]]
            │ │ │ numeric_value_mask (torch.bool):
            │ │ │ │ [[False,  True,  ...,  True, False],
            │ │ │ │  [ True,  True,  ...,  True, False],
            │ │ │ │  [ True,  True,  ...,  True, False],
            │ │ │ │  [ True,  True,  ...,  True, False]]
            │ │
            │ │ Static:
            │ │ │ static_code (torch.int64):
            │ │ │ │ [[7, 9],
            │ │ │ │  [6, 9],
            │ │ │ │  [8, 9],
            │ │ │ │  [8, 9]]
            │ │ │ static_numeric_value (torch.float32):
            │ │ │ │ [[ 0.00,  1.58],
            │ │ │ │  [ 0.00,  0.07],
            │ │ │ │  [ 0.00, -0.54],
            │ │ │ │  [ 0.00, -1.10]]
            │ │ │ static_numeric_value_mask (torch.bool):
            │ │ │ │ [[False,  True],
            │ │ │ │  [False,  True],
            │ │ │ │  [False,  True],
            │ │ │ │  [False,  True]]
            >>> sample_pytorch_dataset.config.padding_side = "right"
            >>> raw_batch = [sample_pytorch_dataset[i] for i in range(len(sample_pytorch_dataset))]
            >>> print(sample_pytorch_dataset.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Measurement (SM)
            │ Static data? ✓
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 4
            │ │ Sequence length: 10
            │ │
            │ │ All dynamic data: (4, 10)
            │ │ Static data: (4, 2)
            │
            │ Data:
            │ │ Dynamic:
            │ │ │ time_delta_days (torch.float32):
            │ │ │ │ [[1.07e+04, 0.00e+00,  ..., 0.00e+00, 2.08e-02],
            │ │ │ │  [0.00e+00, 1.37e-02,  ..., 0.00e+00, 5.91e-03],
            │ │ │ │  [0.00e+00, 1.18e+04,  ..., 0.00e+00, 0.00e+00],
            │ │ │ │  [0.00e+00, 1.24e+04,  ..., 0.00e+00, 0.00e+00]]
            │ │ │ code (torch.int64):
            │ │ │ │ [[ 1, 10,  ..., 11,  4],
            │ │ │ │  [11, 10,  ..., 11,  4],
            │ │ │ │  [ 5,  3,  ...,  0,  0],
            │ │ │ │  [ 5,  2,  ...,  0,  0]]
            │ │ │ numeric_value (torch.float32):
            │ │ │ │ [[ 0.00, -0.57,  ..., -1.53,  0.00],
            │ │ │ │  [ 0.80,  0.34,  ...,  1.00,  0.00],
            │ │ │ │  [ 0.00,  0.00,  ...,  0.00,  0.00],
            │ │ │ │  [ 0.00,  0.00,  ...,  0.00,  0.00]]
            │ │ │ numeric_value_mask (torch.bool):
            │ │ │ │ [[False,  True,  ...,  True, False],
            │ │ │ │  [ True,  True,  ...,  True, False],
            │ │ │ │  [False, False,  ...,  True,  True],
            │ │ │ │  [False, False,  ...,  True,  True]]
            │ │
            │ │ Static:
            │ │ │ static_code (torch.int64):
            │ │ │ │ [[7, 9],
            │ │ │ │  [6, 9],
            │ │ │ │  [8, 9],
            │ │ │ │  [8, 9]]
            │ │ │ static_numeric_value (torch.float32):
            │ │ │ │ [[ 0.00,  1.58],
            │ │ │ │  [ 0.00,  0.07],
            │ │ │ │  [ 0.00, -0.54],
            │ │ │ │  [ 0.00, -1.10]]
            │ │ │ static_numeric_value_mask (torch.bool):
            │ │ │ │ [[False,  True],
            │ │ │ │  [False,  True],
            │ │ │ │  [False,  True],
            │ │ │ │  [False,  True]]

            Static data can also be omitted if set in the config.

            >>> sample_pytorch_dataset.config.static_inclusion_mode = StaticInclusionMode.OMIT
            >>> sample_pytorch_dataset.config.seq_sampling_strategy = SubsequenceSamplingStrategy.RANDOM
            >>> raw_batch = [sample_pytorch_dataset[2], sample_pytorch_dataset[3]]
            >>> print(sample_pytorch_dataset.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Measurement (SM)
            │ Static data? ✗
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 2
            │ │ Sequence length: 5
            │ │
            │ │ All dynamic data: (2, 5)
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

            Static data can also be prepended to the dynamic data.

            >>> sample_pytorch_dataset.config.static_inclusion_mode = StaticInclusionMode.PREPEND
            >>> sample_pytorch_dataset.config.seq_sampling_strategy = SubsequenceSamplingStrategy.TO_END
            >>> raw_batch = [sample_pytorch_dataset[2], sample_pytorch_dataset[3]]
            >>> print(sample_pytorch_dataset.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Measurement (SM)
            │ Static data? ✓ (prepended)
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 2
            │ │ Sequence length (static + dynamic): 7
            │ │
            │ │ All [static; dynamic] data: (2, 7)
            │
            │ Data:
            │ │ [Static; Dynamic]:
            │ │ │ time_delta_days (torch.float32):
            │ │ │ │ [[0.00, 0.00,  ..., 0.00, 0.10],
            │ │ │ │  [0.00, 0.00,  ..., 0.00, 0.05]]
            │ │ │ code (torch.int64):
            │ │ │ │ [[ 8,  9,  ..., 11,  4],
            │ │ │ │  [ 8,  9,  ..., 11,  4]]
            │ │ │ numeric_value (torch.float32):
            │ │ │ │ [[ 0.00, -0.54,  ..., -0.34,  0.00],
            │ │ │ │  [ 0.00, -1.10,  ...,  0.85,  0.00]]
            │ │ │ numeric_value_mask (torch.bool):
            │ │ │ │ [[False,  True,  ...,  True, False],
            │ │ │ │  [False,  True,  ...,  True, False]]
            │ │ │ static_mask (torch.bool):
            │ │ │ │ [[ True,  True,  ..., False, False],
            │ │ │ │  [ True,  True,  ..., False, False]]

            If the batch mode is SEM, the event mask will also be included and the output shape will differ:

            >>> sample_pytorch_dataset.config.batch_mode = "SEM"
            >>> sample_pytorch_dataset.config.static_inclusion_mode = StaticInclusionMode.OMIT
            >>> raw_batch = [sample_pytorch_dataset[2], sample_pytorch_dataset[3]]
            >>> print(sample_pytorch_dataset.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Event-Measurement (SEM)
            │ Static data? ✗
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 2
            │ │ Sequence length: 3
            │ │ Event length: 3
            │ │
            │ │ Per-event data: (2, 3)
            │ │ Per-measurement data: (2, 3, 3)
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

            Padding side changes work in this mode as well.

            >>> sample_pytorch_dataset.config.padding_side = "left"
            >>> print(sample_pytorch_dataset.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Event-Measurement (SEM)
            │ Static data? ✗
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 2
            │ │ Sequence length: 3
            │ │ Event length: 3
            │ │
            │ │ Per-event data: (2, 3)
            │ │ Per-measurement data: (2, 3, 3)
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
            │ │ │ │ [[[ 0,  0,  5],
            │ │ │ │   [ 3, 10, 11],
            │ │ │ │   [ 0,  0,  4]],
            │ │ │ │  [[ 0,  0,  5],
            │ │ │ │   [ 2, 10, 11],
            │ │ │ │   [ 0,  0,  4]]]
            │ │ │ numeric_value (torch.float32):
            │ │ │ │ [[[ 0.00,  0.00,  0.00],
            │ │ │ │   [ 0.00, -1.45, -0.34],
            │ │ │ │   [ 0.00,  0.00,  0.00]],
            │ │ │ │  [[ 0.00,  0.00,  0.00],
            │ │ │ │   [ 0.00,  3.00,  0.85],
            │ │ │ │   [ 0.00,  0.00,  0.00]]]
            │ │ │ numeric_value_mask (torch.bool):
            │ │ │ │ [[[ True,  True, False],
            │ │ │ │   [False,  True,  True],
            │ │ │ │   [ True,  True, False]],
            │ │ │ │  [[ True,  True, False],
            │ │ │ │   [False,  True,  True],
            │ │ │ │   [ True,  True, False]]]

            In this mode, though redundant, the static mask will still be present if static data is prepended

            >>> sample_pytorch_dataset.config.batch_mode = "SEM"
            >>> sample_pytorch_dataset.config.padding_side = "right"
            >>> sample_pytorch_dataset.config.static_inclusion_mode = StaticInclusionMode.PREPEND
            >>> sample_pytorch_dataset.config.seq_sampling_strategy = SubsequenceSamplingStrategy.TO_END
            >>> raw_batch = [sample_pytorch_dataset[2], sample_pytorch_dataset[3]]
            >>> print(sample_pytorch_dataset.collate(raw_batch))
            MEDSTorchBatch:
            │ Mode: Subject-Event-Measurement (SEM)
            │ Static data? ✓ (prepended)
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 2
            │ │ Sequence length (static + dynamic): 4
            │ │ Event length: 3
            │ │
            │ │ Per-event data: (2, 4)
            │ │ Per-measurement data: (2, 4, 3)
            │
            │ Data:
            │ │ Event-level:
            │ │ │ time_delta_days (torch.float32):
            │ │ │ │ [[0.00e+00, 0.00e+00, 1.18e+04, 9.79e-02],
            │ │ │ │  [0.00e+00, 0.00e+00, 1.24e+04, 4.64e-02]]
            │ │ │ event_mask (torch.bool):
            │ │ │ │ [[True, True, True, True],
            │ │ │ │  [True, True, True, True]]
            │ │ │ static_mask (torch.bool):
            │ │ │ │ [[ True, False, False, False],
            │ │ │ │  [ True, False, False, False]]
            │ │
            │ │ Measurement-level:
            │ │ │ code (torch.int64):
            │ │ │ │ [[[ 8,  9,  0],
            │ │ │ │   [ 5,  0,  0],
            │ │ │ │   [ 3, 10, 11],
            │ │ │ │   [ 4,  0,  0]],
            │ │ │ │  [[ 8,  9,  0],
            │ │ │ │   [ 5,  0,  0],
            │ │ │ │   [ 2, 10, 11],
            │ │ │ │   [ 4,  0,  0]]]
            │ │ │ numeric_value (torch.float32):
            │ │ │ │ [[[ 0.00, -0.54,  0.00],
            │ │ │ │   [ 0.00,  0.00,  0.00],
            │ │ │ │   [ 0.00, -1.45, -0.34],
            │ │ │ │   [ 0.00,  0.00,  0.00]],
            │ │ │ │  [[ 0.00, -1.10,  0.00],
            │ │ │ │   [ 0.00,  0.00,  0.00],
            │ │ │ │   [ 0.00,  3.00,  0.85],
            │ │ │ │   [ 0.00,  0.00,  0.00]]]
            │ │ │ numeric_value_mask (torch.bool):
            │ │ │ │ [[[False,  True,  True],
            │ │ │ │   [False,  True,  True],
            │ │ │ │   [False,  True,  True],
            │ │ │ │   [False,  True,  True]],
            │ │ │ │  [[False,  True,  True],
            │ │ │ │   [False,  True,  True],
            │ │ │ │   [False,  True,  True],
            │ │ │ │   [False,  True,  True]]]
        """

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch])
        data = data.to_dense(padding_side=self.config.padding_side)
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}

        out = {}
        out["time_delta_days"] = torch.nan_to_num(tensorized.pop("time_delta_days"), nan=0).float()
        out["code"] = tensorized.pop("code").long()
        if self.config.batch_mode == BatchMode.SEM:
            out["event_mask"] = tensorized.pop("dim1/mask")
        out["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()
        out["numeric_value_mask"] = ~torch.isnan(tensorized.pop("numeric_value"))

        match self.config.static_inclusion_mode:
            case StaticInclusionMode.OMIT:
                pass
            case StaticInclusionMode.INCLUDE:
                static_data = JointNestedRaggedTensorDict(
                    {
                        "static_code": [item["static_code"] for item in batch],
                        "static_numeric_value": [item["static_numeric_value"] for item in batch],
                    }
                ).to_dense()
                static_tensorized = {k: torch.as_tensor(v) for k, v in static_data.items()}
                out["static_code"] = static_tensorized.pop("static_code").long()
                out["static_numeric_value"] = torch.nan_to_num(
                    static_tensorized["static_numeric_value"], nan=0
                ).float()
                out["static_numeric_value_mask"] = ~torch.isnan(static_tensorized["static_numeric_value"])
            case StaticInclusionMode.PREPEND:
                n_static_measurements = [item["n_static_measurements"] for item in batch]

                match self.config.batch_mode:
                    case BatchMode.SEM:
                        static_mask = torch.zeros_like(out["event_mask"])
                        static_mask[:, 0] = True
                    case BatchMode.SM:
                        static_mask = torch.arange(out["time_delta_days"].shape[1]).unsqueeze(
                            0
                        ) < torch.as_tensor(n_static_measurements).unsqueeze(1)
                        static_mask = static_mask.to(
                            device=out["numeric_value_mask"].device,
                            dtype=out["numeric_value_mask"].dtype,
                        )

                out["static_mask"] = static_mask

        if self.has_task_labels:
            out[self.LABEL_COL] = torch.Tensor([item[self.LABEL_COL] for item in batch]).bool()

        return MEDSTorchBatch(**out)

    def get_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Constructs a PyTorch DataLoader for this dataset using the dataset's custom collate function.

        Args:
            **kwargs: Additional arguments to pass to the DataLoader constructor.

        Returns:
            torch.utils.data.DataLoader: A DataLoader object for this dataset.

        Examples:
            >>> from meds_torchdata.types import SubsequenceSamplingStrategy
            >>> sample_pytorch_dataset.config.static_inclusion_mode = StaticInclusionMode.INCLUDE
            >>> sample_pytorch_dataset.config.seq_sampling_strategy = SubsequenceSamplingStrategy.TO_END
            >>> sample_pytorch_dataset.config.batch_mode = "SM"
            >>> _ = torch.manual_seed(0)
            >>> torch.use_deterministic_algorithms(True)
            >>> DL = sample_pytorch_dataset.get_dataloader(batch_size=2, shuffle=False)
            >>> print(next(iter(DL)))
            MEDSTorchBatch:
            │ Mode: Subject-Measurement (SM)
            │ Static data? ✓
            │ Labels? ✗
            │
            │ Shape:
            │ │ Batch size: 2
            │ │ Sequence length: 10
            │ │
            │ │ All dynamic data: (2, 10)
            │ │ Static data: (2, 2)
            │
            │ Data:
            │ │ Dynamic:
            │ │ │ time_delta_days (torch.float32):
            │ │ │ │ [[1.07e+04, 0.00e+00,  ..., 0.00e+00, 2.08e-02],
            │ │ │ │  [0.00e+00, 1.37e-02,  ..., 0.00e+00, 5.91e-03]]
            │ │ │ code (torch.int64):
            │ │ │ │ [[ 1, 10,  ..., 11,  4],
            │ │ │ │  [11, 10,  ..., 11,  4]]
            │ │ │ numeric_value (torch.float32):
            │ │ │ │ [[ 0.00, -0.57,  ..., -1.53,  0.00],
            │ │ │ │  [ 0.80,  0.34,  ...,  1.00,  0.00]]
            │ │ │ numeric_value_mask (torch.bool):
            │ │ │ │ [[False,  True,  ...,  True, False],
            │ │ │ │  [ True,  True,  ...,  True, False]]
            │ │
            │ │ Static:
            │ │ │ static_code (torch.int64):
            │ │ │ │ [[7, 9],
            │ │ │ │  [6, 9]]
            │ │ │ static_numeric_value (torch.float32):
            │ │ │ │ [[0.00, 1.58],
            │ │ │ │  [0.00, 0.07]]
            │ │ │ static_numeric_value_mask (torch.bool):
            │ │ │ │ [[False,  True],
            │ │ │ │  [False,  True]]
        """
        return torch.utils.data.DataLoader(self, collate_fn=self.collate, **kwargs)
