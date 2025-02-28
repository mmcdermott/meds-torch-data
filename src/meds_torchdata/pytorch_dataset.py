import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .config import (
    MEDSTorchBatch,
    MEDSTorchDataConfig,
    StaticInclusionMode,
    SubsequenceSamplingStrategy,
)

logger = logging.getLogger(__name__)


BINARY_LABEL_COL = "boolean_value"


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
        TODO

    Examples:
    """

    @staticmethod
    def get_task_indices_and_labels(
        task_df: pl.DataFrame, static_dfs: dict[str, pl.DataFrame]
    ) -> tuple[list[tuple[int, int, int]], dict[str, list]]:
        """Processes the joint DataFrame to determine the index range for each subject's task.

        For each row in task_df_joint, it is assumed that `time` is a sorted column and the function
        computes the index of the last event at `prediction_time`.

        Parameters:
            - task_df_joint (DataFrame): A DataFrame resulting from the merge_task_with_static function.

        Returns:
            - list: list of index tuples of format (subject_id, start_idx, end_idx).
            - dict: dictionary of task names to lists of labels in the same order as the indexes.

        Examples:
        """

        static_df = pl.concat(static_dfs.values()).select("subject_id", "time")

        end_idx_expr = (
            (pl.col("time").search_sorted(pl.col("prediction_time"), side="right")).last().alias("end_idx")
        )

        label_df = (
            task_df.join(static_df, on="subject_id", how="inner")
            .with_row_index("_row_index")
            .explode("time")
            .group_by("_row_index", "subject_id", "prediction_time", "boolean_value", maintain_order=True)
            .agg(end_idx_expr)
        )

        indexes = list(zip(label_df["subject_id"], label_df["end_idx"]))
        labels = label_df[BINARY_LABEL_COL].to_list()
        prediction_times = label_df["prediction_time"].to_list()

        return indexes, labels, prediction_times

    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

        logger.info("Reading subject schema and static data")
        self.read_subject_descriptors()

    def read_subject_descriptors(self):
        """Read subject schemas and static data from the dataset.

        This method processes the Parquet files for each shard in the dataset, extracting static data and
        creating various mappings and indices for efficient data access.

        The method populates the following instance attributes:
        - self.static_dfs: Dictionary of static DataFrames for each shard.
        - self.subj_indices: Mapping of subject IDs to their indices.
        - self.subj_seq_bounds: Dictionary of sequence bounds for each subject.
        - self.index: List of (subject_id, start, end) tuples for data access.
        - self.labels: Dictionary of task labels (if tasks are specified).

        If a task is specified in the configuration, this method also processes the task labels and
        integrates them with the static data.

        Raises:
            ValueError: If duplicate subjects are found across shards.
            FileNotFoundError: If specified task files are not found.
        """

        schema_dir = self.config.tensorized_cohort_dir / "tokenization" / "schemas"

        self.static_dfs = {}
        self.subj_indices = {}
        self.subj_seq_bounds = {}
        self.subj_map = {}

        schema_files = list(schema_dir.glob(f"{self.split}/*.parquet"))
        if not schema_files:
            raise FileNotFoundError(
                f"No schema files found in {schema_dir}! If your data is not sharded by split, this error "
                "may occur because this codebase does not handle non-split sharded data. See Issue #79 for "
                "tracking this issue."
            )

        for schema_fp in schema_files:
            shard = str(schema_fp.relative_to(schema_dir).with_suffix(""))

            df = (
                pl.read_parquet(
                    schema_fp,
                    columns=[
                        "subject_id",
                        "start_time",
                        "code",
                        "numeric_value",
                        "time",
                    ],
                    use_pyarrow=True,
                )
                .rename({"code": "static_indices", "numeric_value": "static_values"})
                .with_columns(
                    pl.col("static_values").list.eval(pl.element().fill_null(0)),
                    pl.col("static_indices").list.eval(pl.element().fill_null(0)),
                )
            )

            self.static_dfs[shard] = df
            subject_ids = df["subject_id"]
            self.subj_map.update({subj: shard for subj in subject_ids})

            n_events = df.select(pl.col("time").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events_count) in enumerate(zip(subject_ids, n_events)):
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events_count)

        if self.has_task:
            task_df_fp = Path(self.config.task_label_path)
            if not task_df_fp.is_file():
                logger.info(f"If the task file is not found at {task_df_fp}")
                task_df_fp = task_df_fp.with_suffix("") / "**/*.parquet"
                logger.info(f"Searching for task parquets over the glob {task_df_fp}")

            logger.info(f"Reading task constraints for {self.config.task_name} from {task_df_fp}")
            task_df = pl.read_parquet(task_df_fp)

            (
                subjs_and_ends,
                self.labels,
                self.prediction_times,
            ) = MEDSPytorchDataset.get_task_indices_and_labels(task_df, self.static_dfs)
            self.index = [(subj, 0, end) for subj, end in subjs_and_ends]
        else:
            self.index = [(subj, *bounds) for subj, bounds in self.subj_seq_bounds.items()]
            self.labels = None

    @property
    def subject_ids(self) -> list[int]:
        return [x[0] for x in self.index]

    def __len__(self):
        return len(self.index)

    @property
    def has_task(self) -> bool:
        return self.config.task_labels_dir is not None

    @property
    def max_seq_len(self) -> int:
        return self.config.max_seq_len

    @property
    def static_inclusion_mode(self) -> StaticInclusionMode:
        return self.config.static_inclusion_mode

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset.

        This method returns a dictionary corresponding to a single subject's data at the specified index. The
        data is not tensorized in this method, as that work is typically done in the collate function.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the data for the specified index. The structure typically includes:
                - code: List of categorical metadata elements.
                - mask: Mask of valid elements in the sequence, False means it is a padded element.
                - numeric_value: List of dynamic numeric values.
                - numeric_value_mask: Mask of numeric values (False means no numeric value was recorded)
                - time_delta_days: List of dynamic time deltas between observations.
                - static_indices(Optional): List of static MEDS codes.
                - static_values(Optional): List of static MEDS numeric values.
                - static_mask(Optional): List of static masks (True means the value is static).

        """
        subject_id, st, end = self.index[idx]
        dynamic_data = self.load_subject_dynamic_data(subject_id, st, end)

        out = self.load_subject(dynamic_data, subject_id, st, end)

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id
        if self.config.do_include_prediction_time:
            if not self.has_task:
                if not self.config.do_include_end_time:
                    raise ValueError(
                        "Cannot include prediction_time without a task specified or do_include_end_time!"
                    )
                else:
                    out["prediction_time"] = out["end_time"]
            else:
                out["prediction_time"] = self.prediction_times[idx]

        if self.labels is not None:
            out[BINARY_LABEL_COL] = self.labels[idx]

        return out

    def load_subject_dynamic_data(
        self, subject_id: int, st: int | None, end: int | None
    ) -> JointNestedRaggedTensorDict:
        """Loads and returns the dynamic data slice for a given subject ID and permissible event range.

        Args:
            subject_id: The ID of the subject to load.
            st: The (integral) index of the first permissible event (meaning unique timestamp) that can be
                read for this subject's record. If None, no limit is applied.
            end: The (integral) index of the last permissible event (meaning unique timestamp) that can be
                 read for this subject's record. If None, no limit is applied.

        Returns:
            A `JointNestedRaggedTensorDict` object containing the dynamic data for the permissible range for
            the given subject.
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]

        dynamic_data_fp = self.config.tensorized_cohort_dir / "data" / f"{shard}.nrt"

        subject_dynamic_data = JointNestedRaggedTensorDict(tensors_fp=dynamic_data_fp)[subject_idx, st:end]

        return subject_dynamic_data

    def load_subject(
        self, subject_dynamic_data, subject_id: int, global_st: int, global_end: int
    ) -> dict[str, list[float]]:
        """Load and process data for a single subject.

        Args:
            subject_dynamic_data: The dynamic data for the subject.
            subject_id (int): The ID of the subject to load.
            global_st (int): The start index of the sequence to load.
            global_end (int): The end index of the sequence to load.

        Returns:
            dict: A dictionary containing the processed data for the subject.

        Examples:
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()

        max_seq_len = self.config.max_seq_len

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_values": static_row["static_values"].item().to_list(),
        }

        if self.static_inclusion_mode == StaticInclusionMode.PREPEND:
            n_static = len(out["static_indices"])
            if n_static >= max_seq_len:
                raise ValueError(
                    f"Static data length {n_static} matches or exceeds "
                    f"max_seq_len {max_seq_len} for subject {subject_id}!"
                )

            max_seq_len -= n_static

        if self.config.do_flatten_tensors:
            # Store original lengths for each time step before flattening
            cum_lens = subject_dynamic_data.tensors["dim1/bounds"]
            subject_dynamic_data = subject_dynamic_data.flatten()

        seq_len = len(subject_dynamic_data)
        st_offset = SubsequenceSamplingStrategy.subsample_st_offset(
            self.config.seq_sampling_strategy, seq_len, max_seq_len
        )
        if st_offset is None:
            st_offset = 0

        end = min(seq_len, st_offset + max_seq_len)
        subject_dynamic_data = subject_dynamic_data[st_offset:end]

        if self.config.do_flatten_tensors:
            # Map flattened indices back to original time indices
            new_global_st = global_st + np.searchsorted(cum_lens, st_offset, side="right").item()
            new_global_end = global_st + np.searchsorted(cum_lens, end, side="right").item()
        else:
            new_global_st = global_st + st_offset
            new_global_end = new_global_st + len(subject_dynamic_data)

        global_st = new_global_st
        global_end = new_global_end

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = global_st
            out["end_idx"] = global_end

        tensors = subject_dynamic_data.tensors

        if self.static_inclusion_mode == StaticInclusionMode.PREPEND:
            tensors["dim0/time_delta_days"] = np.concatenate(
                [np.zeros(len(out["static_indices"])), tensors["dim0/time_delta_days"]]
            )
            tensors["dim0/static_mask"] = np.concatenate(
                [
                    np.ones(len(out["static_indices"]), dtype=bool),
                    np.zeros(len(tensors["dim0/code"]), dtype=bool),
                ]
            )
            tensors["dim0/code"] = np.concatenate([out["static_indices"], tensors["dim0/code"]])
            tensors["dim0/numeric_value"] = np.concatenate(
                [out["static_values"], tensors["dim0/numeric_value"]]
            )
        else:
            tensors["dim0/static_mask"] = np.zeros(len(tensors["dim0/code"]), dtype=bool)

        subject_dynamic_data = JointNestedRaggedTensorDict(processed_tensors=tensors)

        out["dynamic"] = subject_dynamic_data

        if self.config.do_include_start_time:
            out["start_time"] = static_row["time"].item().to_list()[global_st]
        if self.config.do_include_end_time:
            out["end_time"] = static_row["time"].item().to_list()[global_end - 1]

        return out

    def collate(self, batch: list[dict]) -> MEDSTorchBatch:
        """Combines a batch of data points into a single, tensorized batch.

        The collated output is a fully tensorized and padded dictionary, ready for input into an
        `input_encoder`. This method uses the JointNestedRaggedTensorDict API to collate and pad the data.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single sample as
                returned by the __getitem__ method.

        Returns:
            dict: A dictionary containing the collated batch data.
        """

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch]).to_dense()
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}
        tensorized["code"] = tensorized["code"].long()
        tensorized["mask"] = tensorized.pop("dim1/mask")
        tensorized["numeric_value_mask"] = ~torch.isnan(tensorized["numeric_value"])
        tensorized["time_delta_days"] = torch.nan_to_num(tensorized["time_delta_days"], nan=0).float()
        tensorized["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()

        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices", "static_mask"):
                if isinstance(batch[0][k], datetime):
                    tensorized[k] = [item[k] for item in batch]
                else:
                    tensorized[k] = torch.Tensor([item[k] for item in batch])
        return tensorized

    def get_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Constructs a PyTorch DataLoader for this dataset.

        Args:
            **kwargs: Additional arguments to pass to the DataLoader constructor.

        Returns:
            torch.utils.data.DataLoader: A DataLoader object for this dataset.
        """
        return torch.utils.data.DataLoader(self, collate_fn=self.collate, **kwargs)
