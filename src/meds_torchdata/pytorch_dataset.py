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
from .types import BatchMode, MEDSTorchBatch, StaticData, SubsequenceSamplingStrategy

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
    LAST_TIME = "window_last_observed"

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

        # Only read the columns this dataset actually needs. Parquet is columnar so this is
        # a per-column I/O saving at subject-schema load time — most importantly, the
        # `measurements_per_event` list column is only needed by STEP_THROUGH sampling in
        # SM mode (where the expansion uses it to map measurement-level window ends back
        # to event-level indices), so every other config skips it. `start_time` is emitted
        # by preprocessing but never consumed downstream, so it's always skipped.
        needed_schema_cols = [
            DataSchema.subject_id_name,
            DataSchema.time_name,
            "static_code",
            "static_numeric_value",
        ]
        needs_meas_per_event = (
            self.config.seq_sampling_strategy == SubsequenceSamplingStrategy.STEP_THROUGH
            and self.config.batch_mode == BatchMode.SM
        )
        if needs_meas_per_event:
            needed_schema_cols.append("measurements_per_event")

        for shard, schema_fp in self.config.schema_fps:
            if not shard.startswith(f"{self.split}/"):
                continue

            # Inspect the parquet schema first so that older tensorized cohorts missing the
            # `measurements_per_event` column (added when STEP_THROUGH SM mode landed) fail
            # with a clear "re-run preprocessing" error instead of a low-level polars /
            # pyarrow column-not-found traceback. Only the column *we asked for* matters,
            # so we only check when the user's config actually needs it.
            if needs_meas_per_event:
                available = set(pq.read_schema(schema_fp).names)
                if "measurements_per_event" not in available:
                    raise ValueError(
                        f"STEP_THROUGH sampling in SM mode requires the "
                        f"`measurements_per_event` column on the schema parquet at "
                        f"{schema_fp}, which older tensorized cohorts (preprocessed before "
                        "this feature landed) do not have. Re-run preprocessing "
                        "(`MTD_preprocess`) to produce it."
                    )

            df = pl.read_parquet(schema_fp, columns=needed_schema_cols, use_pyarrow=True).with_columns(
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
            zip(self.schema_df[DataSchema.subject_id_name], self.schema_df[self.END_IDX], strict=True)
        )
        self.labels = self.schema_df[self.LABEL_COL] if self.has_task_labels else None

        # STEP_THROUGH state:
        # - `_windows_per_subject`: `subject_id -> number of dataset elements the subject
        #   expands into`, populates `MEDSTorchBatch.n_subject_windows` when
        #   `config.include_subject_window_counts_in_batch` is set.
        # - `step_through_meas_ends`: parallel list to `self.index`, only populated in
        #   `BatchMode.SM` step-through. Stores the measurement-level end of each window so
        #   `process_dynamic_data` can slice mid-event via its `explicit_end` kwarg. `None`
        #   in SEM mode because the event-level `end` in `self.index` plus TO_END sampling
        #   is sufficient there.
        self._windows_per_subject: dict[int, int] | None = None
        self.step_through_meas_ends: list[int] | None = None
        if self.config.seq_sampling_strategy == SubsequenceSamplingStrategy.STEP_THROUGH:
            self._expand_index_for_step_through()

    def _expand_index_for_step_through(self) -> None:
        """Expand `self.index` so that STEP_THROUGH sampling produces one entry per window.

        For each subject in the pre-expansion index, this walks a sliding window of size
        `max_seq_len` across the permitted sequence with either a user-supplied
        `step_through_stride` or a user-supplied `step_through_overlap`, producing one
        dataset element per window.

        The walk is expressed in the same unit as `max_seq_len`: events in `BatchMode.SEM`,
        measurements in `BatchMode.SM`. In SM mode this means the window ends can fall
        mid-event — for example a subject with two events [3, 5] measurements and
        `max_seq_len=4` with `step_through_stride=2` produces windows `[0:4)`, `[2:6)`, and
        `[4:8)` — the second window ends in the middle of the second event. This is the
        intentional Design B semantics: step-through walks the **measurement-level** sequence
        regardless of event atomicity. See the class docstring for alternatives.

        The per-subject measurement-level walk is powered by `measurements_per_event`, a new
        preprocessing column that records the measurement count at each unique timestamp for
        each subject. In SM mode we use `np.searchsorted` on the per-subject cumulative sum
        to find the smallest event index whose prefix contains each target measurement end —
        that's the `end` stored in `self.index` (for `load_subject_data`) — while the
        measurement-level end itself is recorded in `self.step_through_meas_ends` and passed
        through `MEDSTorchDataConfig.process_dynamic_data`'s `explicit_end` kwarg at sample
        time.

        Validation at construction time:
        - `stride` (or the derived `effective_window - overlap`) must be positive.
        - `stride <= effective_window` per subject, so consecutive windows overlap by
          `effective_window - stride >= 0` elements and no data is skipped.
        - For SM mode, the `measurements_per_event` column must exist on the schema parquet
          (re-run preprocessing if it's missing from an older cohort).

        A warning with observed expansion stats is logged on startup; set
        `config.include_subject_window_counts_in_batch=True` to surface per-sample window
        counts in the collated batch so downstream code can reweight losses.

        Examples:
            Example 1 — SEM mode, event-level walk:

            >>> import dataclasses
            >>> cfg = dataclasses.replace(
            ...     sample_dataset_config,
            ...     max_seq_len=3,
            ...     seq_sampling_strategy="step_through",
            ...     step_through_stride=2,
            ...     batch_mode="SEM",
            ...     static_inclusion_mode="omit",
            ...     include_subject_window_counts_in_batch=True,
            ... )
            >>> pyd = MEDSPytorchDataset(cfg, split="train")

            The four subjects in the fixture have event counts of 6, 8, 3, and 3. With
            `max_seq_len=3, step_through_stride=2`, `self.index` has one entry per window —
            each entry's `end` is the *window*'s final event. `self.step_through_meas_ends`
            stays `None` in SEM mode because the sampler's `TO_END` semantics handle the
            window end natively via event-level slicing.

            >>> pyd.index
            [(239684, 3), (239684, 5), (239684, 6), (1195293, 3), (1195293, 5), (1195293, 7),
             (1195293, 8), (68729, 3), (814703, 3)]
            >>> pyd.step_through_meas_ends is None
            True
            >>> pyd._windows_per_subject
            {239684: 3, 1195293: 4, 68729: 1, 814703: 1}

            Per-sample output is the window and carries the per-subject window count when
            the config flag is set. Sample 0 is subject 239684's first window — three events
            starting at event 0 (note that the subject's static event has been prepended
            into the code vocabulary as event ``5`` during preprocessing):

            >>> sample = pyd[0]
            >>> sample["n_subject_windows"]
            3
            >>> sample["dynamic"].to_dense()["code"]
            array([[ 5,  0,  0],
                   [ 1, 10, 11],
                   [10, 11,  0]], dtype=uint8)

            Collated batches surface `n_subject_windows` as a `[batch_size]` tensor — use
            `1 / n_subject_windows` as a per-sample loss weight to undo oversampling:

            >>> batch = pyd.collate([pyd[0], pyd[1], pyd[7]])
            >>> batch.n_subject_windows
            tensor([3, 3, 1])

            Example 2 — SM mode, measurement-level walk that crosses event boundaries:

            SM mode interprets `max_seq_len` and stride as **measurements**, not events.
            With `max_seq_len=5, step_through_stride=3`, subject 239684 (which has 6 events
            flattening to a total of 11 measurements) produces three windows, each exactly
            5 measurements wide: the first ends at measurement 5, the second at 8, the
            third at the tail (11). The index stores the smallest event index whose prefix
            contains each measurement-level end (used by `load_subject_data`), and
            `self.step_through_meas_ends` stores the actual measurement-level ends that get
            passed through `process_dynamic_data.explicit_end` at sample time:

            >>> sm_cfg = dataclasses.replace(
            ...     sample_dataset_config,
            ...     max_seq_len=5,
            ...     seq_sampling_strategy="step_through",
            ...     step_through_stride=3,
            ...     batch_mode="SM",
            ...     static_inclusion_mode="omit",
            ... )
            >>> sm_pyd = MEDSPytorchDataset(sm_cfg, split="train")
            >>> [entry for entry in sm_pyd.index if entry[0] == 239684]
            [(239684, 3), (239684, 4), (239684, 6)]
            >>> [
            ...     meas_end for (subj, _), meas_end in
            ...     zip(sm_pyd.index, sm_pyd.step_through_meas_ends, strict=True)
            ...     if subj == 239684
            ... ]
            [5, 8, 11]

            **The critical Design B property** — the second window (measurement end 8)
            begins *in the middle of event 2* (events [0:1] contain 1+1=2 measurements,
            event 2 begins at measurement position 2 and contains 3 measurements, so
            measurement 3 is inside event 2). The window is exactly 5 measurements wide
            regardless of where event boundaries fall:

            >>> sm_pyd[1]["dynamic"].to_dense()["code"]
            array([11, 10, 11, 10, 11], dtype=uint8)
            >>> len(sm_pyd[1]["dynamic"])
            5

            And the third window (measurement end 11) is the subject's tail — also exactly
            5 measurements wide:

            >>> sm_pyd[2]["dynamic"].to_dense()["code"]
            array([10, 11, 10, 11,  4], dtype=uint8)
            >>> len(sm_pyd[2]["dynamic"])
            5
        """

        # 1. Compute the stride (possibly per-subject).
        # 2. Walk window ends.
        # 3. Validate stride <= effective_window per subject.
        # 4. Emit the expanded index and (for SM) the parallel measurement-end list.

        n_subjects_before = len(self.index)

        expanded_index: list[tuple[int, int]] = []
        expanded_meas_ends: list[int] = []
        windows_per_subject: dict[int, int] = {}

        for subject_id, end_idx in self.index:
            effective_window = self._effective_max_seq_len_for(subject_id)
            if effective_window <= 0:
                raise ValueError(
                    f"Effective dynamic window size for subject {subject_id} is "
                    f"{effective_window} (max_seq_len={self.config.max_seq_len} minus the "
                    "static elements that will be prepended in PREPEND mode). Increase "
                    "max_seq_len so at least one dynamic element fits after prepending "
                    "static data."
                )

            stride = self._resolve_step_through_stride_for(subject_id, effective_window)
            if stride <= 0:
                # This only happens when `step_through_overlap` is set (it's relative to the
                # per-subject effective window and can produce a non-positive stride if
                # overlap >= effective_window). A plain stride is already validated to be
                # positive at config time.
                raise ValueError(
                    f"step_through_overlap ({self.config.step_through_overlap}) must be "
                    f"strictly less than the effective window width ({effective_window}) "
                    f"for subject {subject_id}; got overlap >= effective window, which "
                    "would produce a non-positive stride. Reduce step_through_overlap or "
                    "increase max_seq_len."
                )
            if stride > effective_window:
                raise ValueError(
                    f"step_through stride ({stride}) exceeds the effective window width "
                    f"({effective_window}) for subject {subject_id}, which would leave gaps "
                    "in coverage. Either reduce step_through_stride or switch to "
                    "step_through_overlap (which is relative to the effective window and "
                    "cannot produce gaps)."
                )

            if self.config.batch_mode == BatchMode.SEM:
                ends = self._step_through_event_ends_sem(end_idx, stride, effective_window)
                windows_per_subject[subject_id] = len(ends)
                for end in ends:
                    expanded_index.append((subject_id, end))
            else:  # SM mode
                meas_ends, event_ends = self._step_through_ends_sm(subject_id, stride, effective_window)
                windows_per_subject[subject_id] = len(meas_ends)
                for event_end, meas_end in zip(event_ends, meas_ends, strict=True):
                    expanded_index.append((subject_id, event_end))
                    expanded_meas_ends.append(meas_end)

        # (Task mode is already rejected at config time, since `task_labels_dir is not None`
        # forces the sampling strategy to `TO_END`. No need to re-check here.)

        self.index = expanded_index
        self._windows_per_subject = windows_per_subject
        self.step_through_meas_ends = expanded_meas_ends if self.config.batch_mode == BatchMode.SM else None

        # Oversampling warning — emitted after the expansion loop so the numbers we report
        # are the actual observed stats rather than a closed-form guess.
        n_elements = len(expanded_index)
        max_windows = max(windows_per_subject.values()) if windows_per_subject else 0
        mean_windows = n_elements / n_subjects_before if n_subjects_before else 0.0
        logger.warning(
            "STEP_THROUGH sampling expanded %d subjects into %d dataset elements "
            "(mean windows per subject=%.1f, max windows per subject=%d). Subjects with "
            "longer dynamic sequences are oversampled relative to shorter ones by a factor "
            "equal to their per-subject window count. To undo the oversampling at loss time, "
            "set MEDSTorchDataConfig.include_subject_window_counts_in_batch=True and use "
            "`1 / batch.n_subject_windows` as a per-sample loss weight.",
            n_subjects_before,
            n_elements,
            mean_windows,
            max_windows,
        )

    def _resolve_step_through_stride_for(self, subject_id: int, effective_window: int) -> int:
        """Return the step-through stride (same unit as `max_seq_len`) for a given subject.

        When `config.step_through_stride` is set directly, that value is used as-is. When
        `config.step_through_overlap` is set instead, the stride is computed relative to the
        per-subject effective window so that consecutive windows share exactly the requested
        overlap regardless of how `PREPEND` shrinks the window for that subject.

        Examples:
            >>> import dataclasses
            >>> stride_cfg = dataclasses.replace(
            ...     sample_dataset_config,
            ...     max_seq_len=3,
            ...     seq_sampling_strategy="step_through",
            ...     step_through_stride=2,
            ...     batch_mode="SEM",
            ...     static_inclusion_mode="omit",
            ... )
            >>> stride_pyd = MEDSPytorchDataset(stride_cfg, split="train")
            >>> stride_pyd._resolve_step_through_stride_for(239684, effective_window=3)
            2

            With `step_through_overlap` the stride varies per subject to honor the
            requested overlap count relative to that subject's effective window:

            >>> overlap_cfg = dataclasses.replace(stride_cfg, step_through_stride=None,
            ...                                   step_through_overlap=1)
            >>> overlap_pyd = MEDSPytorchDataset(overlap_cfg, split="train")
            >>> overlap_pyd._resolve_step_through_stride_for(239684, effective_window=3)
            2
            >>> overlap_pyd._resolve_step_through_stride_for(239684, effective_window=5)
            4
        """

        if self.config.step_through_stride is not None:
            return self.config.step_through_stride
        return effective_window - self.config.step_through_overlap

    @staticmethod
    def _step_through_event_ends_sem(end_idx: int, stride: int, effective_window: int) -> list[int]:
        """Return the list of event-level window ends for a SEM-mode step-through walk.

        The first window ends at `effective_window` (so it contains `effective_window`
        events); subsequent windows each shift forward by `stride` events; the final window
        is anchored to `end_idx` so the last event is always covered regardless of stride.

        Examples:
            Typical overlapping walk: `end_idx=8, stride=2, effective_window=3` produces
            windows ending at events `[3, 5, 7, 8]` — the last one is tail-anchored to
            `end_idx` so the final event is always covered:

            >>> MEDSPytorchDataset._step_through_event_ends_sem(8, stride=2, effective_window=3)
            [3, 5, 7, 8]

            Contiguous (`stride == effective_window`) walk:

            >>> MEDSPytorchDataset._step_through_event_ends_sem(8, stride=3, effective_window=3)
            [3, 6, 8]

            Short subject (`end_idx <= effective_window`) — single window covering everything:

            >>> MEDSPytorchDataset._step_through_event_ends_sem(3, stride=2, effective_window=3)
            [3]
            >>> MEDSPytorchDataset._step_through_event_ends_sem(2, stride=2, effective_window=3)
            [2]

            Stride-divides-gap — no duplicate tail anchor:

            >>> MEDSPytorchDataset._step_through_event_ends_sem(7, stride=2, effective_window=3)
            [3, 5, 7]
        """

        if end_idx <= effective_window:
            return [end_idx]
        ends = list(range(effective_window, end_idx, stride))
        if not ends or ends[-1] != end_idx:
            ends.append(end_idx)
        return ends

    def _step_through_ends_sm(
        self, subject_id: int, stride: int, effective_window: int
    ) -> tuple[list[int], list[int]]:
        """Return measurement- and event-level window ends for an SM-mode step-through walk.

        Walks the measurement-level window ends `[effective_window, effective_window+stride,
        ..., total_meas]` using the per-subject `measurements_per_event` list from the
        schema. Each measurement-level end is converted to the smallest event index whose
        prefix contains it via `np.searchsorted` on the cumulative-measurement array — that
        becomes the `end` the loader reads from `self.index`, while the measurement-level
        end is returned separately for `self.step_through_meas_ends`.

        Examples:
            Subject 239684 in the `sample_dataset_config` fixture has 6 events flattening
            to 11 measurements with per-event counts `[1, 3, 2, 2, 2, 1]`, so
            `cum_meas = [0, 1, 4, 6, 8, 10, 11]`. With `effective_window=5, stride=3`, the
            walk produces measurement ends `[5, 8, 11]`, each of which maps via
            `searchsorted(cum_meas, meas_end, side="left")` to the smallest event index
            whose prefix contains it. Note that window 2 (meas end `8`) maps to event `4`
            because `cum_meas[4] == 8` exactly, while window 1 (meas end `5`) maps to event
            `3` because `cum_meas[2] = 4 < 5 <= cum_meas[3] = 6`:

            >>> import dataclasses
            >>> sm_cfg = dataclasses.replace(
            ...     sample_dataset_config,
            ...     max_seq_len=5,
            ...     seq_sampling_strategy="step_through",
            ...     step_through_stride=3,
            ...     batch_mode="SM",
            ...     static_inclusion_mode="omit",
            ... )
            >>> sm_pyd = MEDSPytorchDataset(sm_cfg, split="train")
            >>> sm_pyd._step_through_ends_sm(239684, stride=3, effective_window=5)
            ([5, 8, 11], [3, 4, 6])

            Short subject (total measurements `<= effective_window`) — single entry
            covering the entire subject:

            >>> sm_pyd._step_through_ends_sm(239684, stride=3, effective_window=20)
            ([11], [6])
        """

        # `__init__` has already verified that `measurements_per_event` exists on every
        # schema parquet this dataset reads (the check lives there so we can raise a clean
        # "re-run preprocessing" error before the eager `pl.read_parquet(columns=...)`
        # would otherwise blow up with a low-level parquet/column-not-found traceback).
        shard, subject_idx = self.subj_locations[subject_id]
        schema_row = self.schema_dfs_by_shard[shard][subject_idx]
        meas_per_event_series = schema_row["measurements_per_event"].item()
        if meas_per_event_series is None:
            # Subject with no dynamic data — single trivial window.
            return [0], [0]
        meas_per_event = meas_per_event_series.to_list()
        cum_meas = np.cumsum([0, *meas_per_event])
        total_meas = int(cum_meas[-1])

        if total_meas <= effective_window:
            # Subject is shorter than one window — emit a single entry covering everything.
            return [total_meas], [len(meas_per_event)]

        meas_ends = list(range(effective_window, total_meas, stride))
        if not meas_ends or meas_ends[-1] != total_meas:
            meas_ends.append(total_meas)

        event_ends = np.searchsorted(cum_meas, meas_ends, side="left").tolist()
        return meas_ends, [int(e) for e in event_ends]

    def _effective_max_seq_len_for(self, subject_id: int) -> int:
        """Return the dynamic-window size a step-through sample can reserve for this subject.

        This mirrors the `max_seq_len -= n_static_seq_els` adjustment inside
        `MEDSTorchDataConfig.process_dynamic_data` for `PREPEND` mode — so that the
        resulting `[static; dynamic]` sample after prepending still has length
        `<= config.max_seq_len`. For every other static inclusion mode this returns
        `config.max_seq_len` unchanged. In `SM + PREPEND` the reduction varies per subject
        because `n_static_seq_els == len(static_code[subject_id])`.

        Examples:
            With the default `sample_dataset_config` (`max_seq_len=10`, SM batch mode,
            `static_inclusion_mode=INCLUDE`), the effective window equals `max_seq_len`
            unchanged for every subject:

            >>> pyd = sample_pytorch_dataset
            >>> pyd.config.max_seq_len
            10
            >>> pyd.config.batch_mode
            <BatchMode.SM: 'SM'>
            >>> pyd.config.static_inclusion_mode
            <StaticInclusionMode.INCLUDE: 'include'>
            >>> [pyd._effective_max_seq_len_for(s) for s in (239684, 1195293, 68729, 814703)]
            [10, 10, 10, 10]

            In `SM + PREPEND` the reduction is `len(static_code)` per subject — every
            fixture subject has two static codes, so their effective window shrinks from
            `10` to `8`:

            >>> import dataclasses
            >>> sm_prepend_cfg = dataclasses.replace(
            ...     pyd.config, static_inclusion_mode="prepend"
            ... )
            >>> sm_prepend_pyd = MEDSPytorchDataset(sm_prepend_cfg, split="train")
            >>> [sm_prepend_pyd._effective_max_seq_len_for(s) for s in (239684, 1195293)]
            [8, 8]

            In `SEM + PREPEND` the reduction is a flat `1` (one event slot reserved for the
            prepended static event):

            >>> sem_prepend_cfg = dataclasses.replace(
            ...     pyd.config, batch_mode="SEM", static_inclusion_mode="prepend"
            ... )
            >>> sem_prepend_pyd = MEDSPytorchDataset(sem_prepend_cfg, split="train")
            >>> [sem_prepend_pyd._effective_max_seq_len_for(s) for s in (239684, 1195293)]
            [9, 9]
        """

        max_seq_len = self.config.max_seq_len
        if self.config.static_inclusion_mode != StaticInclusionMode.PREPEND:
            return max_seq_len
        if self.config.batch_mode == BatchMode.SEM:
            return max_seq_len - 1
        # SM + PREPEND: the number of reserved slots is the per-subject static measurement
        # count, read from the schema df without loading the dynamic tensors.
        shard, subject_idx = self.subj_locations[subject_id]
        static_code_list = self.schema_dfs_by_shard[shard][subject_idx]["static_code"].item()
        n_static = len(static_code_list) if static_code_list is not None else 0
        return max_seq_len - n_static

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

        base_df = self._all_schemas

        if self.has_task_index:
            df = self.get_task_seq_bounds_and_labels(self.labels_df, base_df)
        else:
            df = base_df.select(
                DataSchema.subject_id_name, pl.col(DataSchema.time_name).list.len().alias(self.END_IDX)
            )

        # `LAST_TIME` reflects the time of the last event the sampler will include. That only
        # makes sense when the sampler deterministically ends at `end_idx - 1`; non-deterministic
        # samplers (RANDOM, BALANCED_RANDOM) may end earlier, so skip the column for them.
        nondeterministic_samplers = {
            SubsequenceSamplingStrategy.RANDOM,
            SubsequenceSamplingStrategy.BALANCED_RANDOM,
        }
        if (
            self.config.include_window_last_observed_in_schema
            and self.has_task_index
            and self.config.seq_sampling_strategy not in nondeterministic_samplers
        ):
            df = (
                df.join(base_df, on=DataSchema.subject_id_name, how="left", maintain_order="left")
                .with_columns(
                    pl.from_epoch(  # This is a polars error where the timestamp was converted to ints...
                        pl.col(DataSchema.time_name).list.get(pl.col(self.END_IDX) - 1),
                        time_unit="us",
                    ).alias(self.LAST_TIME)
                )
                .drop(DataSchema.time_name)
            )

        return df

    @property
    def _all_schemas(self) -> pl.DataFrame:
        """This is a helper for easy access to the full set of schema dataframes for debugging."""

        return pl.concat(
            (
                df.select(DataSchema.subject_id_name, DataSchema.time_name)
                for df in self.schema_dfs_by_shard.values()
            ),
            how="vertical",
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

        This is a wrapper around the core item-retrieval logic that allows for deterministic subsequence
        sampling via an optional random seed.
        """

        subject_id, end_idx = self.index[idx]
        dynamic_data, static_data = self.load_subject_data(subject_id=subject_id, st=0, end=end_idx)

        match self.config.static_inclusion_mode:
            case StaticInclusionMode.OMIT:
                out = {}
                n_static_seq_els = None
            case StaticInclusionMode.INCLUDE:
                n_static_seq_els = None
                out = {
                    "static_code": static_data.code,
                    "static_numeric_value": static_data.numeric_value,
                }
            case StaticInclusionMode.PREPEND:
                n_static_seq_els = len(static_data.code) if self.config.batch_mode == BatchMode.SM else 1
                out = {"n_static_seq_els": n_static_seq_els}

        # STEP_THROUGH in SM mode pre-computes the measurement-level window end for each
        # sample (because events are not atomic in this mode — the window can terminate
        # mid-event). We pass that through `process_dynamic_data.explicit_end`. In every
        # other config (including SEM step-through), the expanded index's `end_event` is
        # all we need: `process_dynamic_data` + the `STEP_THROUGH → TO_END` delegation in
        # `subsample_st_offset` handles the window at the event level.
        explicit_end = self.step_through_meas_ends[idx] if self.step_through_meas_ends is not None else None
        dynamic_data = self.config.process_dynamic_data(
            dynamic_data,
            n_static_seq_els=n_static_seq_els,
            rng=seed,
            explicit_end=explicit_end,
        )

        # Only leak the per-subject window count into the sample dict when the user has
        # explicitly asked for it in the batch — otherwise the sample API would depend on
        # the sampling strategy, which a user reading individual `dataset[idx]` outputs
        # would find surprising. The collator's fallback-to-1 handles non-step-through
        # datasets that still opt into the batch field.
        if self.config.include_subject_window_counts_in_batch and self._windows_per_subject is not None:
            out["n_subject_windows"] = self._windows_per_subject[subject_id]

        if self.config.static_inclusion_mode == StaticInclusionMode.PREPEND:
            static_as_JNRT = static_data.to_JNRT(self.config.batch_mode, dynamic_data.schema)
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
        # `.item()` returns the polars list for a given row. When the dataset has no static
        # data at all, the column may be null (not just an empty list), in which case `.item()`
        # returns `None` and `.to_list()` would raise `AttributeError`. See issue #63.
        static_code_list = subj_schema["static_code"].item()
        static_numeric_value_list = subj_schema["static_numeric_value"].item()
        static_code = static_code_list.to_list() if static_code_list is not None else []
        static_numeric_value = (
            static_numeric_value_list.to_list() if static_numeric_value_list is not None else []
        )

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
        out["code"] = tensorized.pop("code").long()
        if self.config.batch_mode == BatchMode.SEM:
            out["event_mask"] = tensorized.pop("dim1/mask")
        # Dynamic-field omission (issues #46 and #47): when the user opts out via config,
        # drop the corresponding tensors from the batch entirely. Gating these with the
        # same conditional keeps the hot path branch-free for the default (include both).
        if self.config.include_time_delta:
            out["time_delta_days"] = torch.nan_to_num(tensorized.pop("time_delta_days"), nan=0).float()
        if self.config.include_numeric_value:
            out["numeric_value_mask"] = ~torch.isnan(tensorized["numeric_value"])
            out["numeric_value"] = torch.nan_to_num(tensorized.pop("numeric_value"), nan=0).float()

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
                n_static_seq_els = [item["n_static_seq_els"] for item in batch]

                match self.config.batch_mode:
                    case BatchMode.SEM:
                        static_mask = torch.zeros_like(out["event_mask"])
                        static_mask[:, 0] = True
                    case BatchMode.SM:
                        # Use `out["code"]` for the shape / dtype reference rather than one
                        # of the optional numeric/time fields, so that static_mask still
                        # works when `include_numeric_value=False` or
                        # `include_time_delta=False` drops those from the batch.
                        seq_len_axis = out["code"].shape[1]
                        static_mask = torch.arange(seq_len_axis).unsqueeze(0) < torch.as_tensor(
                            n_static_seq_els
                        ).unsqueeze(1)
                        static_mask = static_mask.to(device=out["code"].device, dtype=torch.bool)

                out["static_mask"] = static_mask

        if self.has_task_labels:
            out[self.LABEL_COL] = torch.Tensor([item[self.LABEL_COL] for item in batch]).bool()

        if self.config.include_subject_window_counts_in_batch:
            # For non-step-through datasets every sample corresponds to one window, so the
            # count is simply 1 for every row — still expose it so downstream loss code can
            # treat the field uniformly regardless of sampling mode.
            counts = [item.get("n_subject_windows", 1) for item in batch]
            out["n_subject_windows"] = torch.as_tensor(counts, dtype=torch.long)

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
