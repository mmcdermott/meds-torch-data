"""Contains configuration objects for building a PyTorch dataset from a MEDS dataset.

This module contains configuration objects for building a PyTorch dataset from a MEDS dataset. These include
enumeration objects for categorical options and a general DataClass configuration object for dataset options.
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
from hydra.core.config_store import ConfigStore
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import open_dict

from .types import BatchMode, PaddingSide, StaticInclusionMode, SubsequenceSamplingStrategy

logger = logging.getLogger(__name__)


@dataclass
class MEDSTorchDataConfig:
    """A data class for storing configuration options for building a PyTorch dataset from a MEDS dataset.

    Attributes:
        tensorized_cohort_dir: Path to the root of a tokenized-and-tensorized MEDS cohort
            produced by `MTD_preprocess`. The directory must contain `tokenization/schemas/`
            and `data/` subdirectories.
        max_seq_len: The maximum length (in the batch mode's natural unit — events in SEM
            mode, measurements in SM mode) of sequences yielded from the dataset. Samplers
            that produce fixed-width windows will use this as the width; `RANDOM` and
            `BALANCED_RANDOM` treat it as the upper bound.
        seq_sampling_strategy: The subsequence sampling strategy — one of
            `SubsequenceSamplingStrategy.{RANDOM, BALANCED_RANDOM, TO_END, FROM_START,
            STEP_THROUGH}`. Task-mode datasets (`task_labels_dir` set) are restricted to
            `TO_END`.
        padding_side: Which side of short sequences to pad when collating into a batch
            (`LEFT` or `RIGHT`). Defaults to `RIGHT`; set to `LEFT` for autoregressive
            generation models.
        static_inclusion_mode: How to surface per-subject static measurements in the
            collated batch — `OMIT` (drop), `INCLUDE` (separate `static_code`/`static_*`
            tensors in the batch), or `PREPEND` (concatenate static elements onto the front
            of the dynamic sequence). In `PREPEND` mode the effective dynamic window shrinks
            to leave room for the static elements.
        task_labels_dir: Optional path to a directory of MEDS Label parquet files. When set,
            the dataset yields one sample per (subject, prediction_time) label with the
            sampling strategy fixed to `TO_END`.
        batch_mode: Whether the collated batch keeps the event/measurement structure
            (`SEM` = Subject-Event-Measurement 3D tensors) or flattens measurements into a
            single sequence dimension (`SM` = Subject-Measurement 2D tensors). The unit of
            `max_seq_len` and every step-through parameter follows this choice.
        include_window_last_observed_in_schema: When True, the `schema_df` helper adds a
            `window_last_observed` column containing the timestamp of the last event included
            in each window. Only meaningful for deterministic samplers (`TO_END`,
            `FROM_START`); skipped for the random samplers because they have no deterministic
            last event.
        step_through_stride: Absolute number of sequence elements (events in SEM mode,
            measurements in SM mode) to advance between consecutive `STEP_THROUGH` windows.
            Must be a positive integer; `bool` is explicitly rejected even though it is a
            subclass of `int`. Mutually exclusive with `step_through_overlap`: exactly one
            of the two must be set when `seq_sampling_strategy == STEP_THROUGH`, and both
            must be `None` for every other strategy.
        step_through_overlap: Alternative to `step_through_stride` that specifies the number
            of elements consecutive `STEP_THROUGH` windows should share. Equivalent to
            `stride = effective_window - overlap`, but more convenient when the user wants
            "no overlap" (`overlap=0`) or a fixed overlap because the effective window can
            vary per subject in `SM + PREPEND`. Must be a non-negative integer strictly less
            than the per-subject effective window.
        include_subject_window_counts_in_batch: When True, `MEDSPytorchDataset.collate`
            populates the `n_subject_windows` field of the returned `MEDSTorchBatch` with the
            number of dataset elements each sample's subject expands into. Intended for
            per-sample loss reweighting (`1 / n_subject_windows`) to undo `STEP_THROUGH`
            oversampling — subjects with longer sequences get expanded into more windows, so
            reweighting by the inverse count restores a per-subject uniform loss.

    Raises:
        FileNotFoundError: If the task_labels_dir or the tensorized_cohort_dir is not a valid directory.
        ValueError: If the subsequence sampling strategy or static inclusion mode is not valid.
        ValueError: If the task_labels_dir is specified but the subsequence sampling strategy is not TO_END.
        ValueError: If `step_through_stride` / `step_through_overlap` is set without the
            `STEP_THROUGH` strategy, if both are set simultaneously, or if either is an
            invalid type (including `bool`) or out of range.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir: # No error
        ...     cfg = MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=tmpdir,
        ...         max_seq_len=10,
        ...     )
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_root = Path(tmpdir) / "tensorized"
        ...     cohort_root.mkdir()
        ...     task_labels_dir = Path(tmpdir) / "task_labels"
        ...     task_labels_dir.mkdir()
        ...     cfg = MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=cohort_root,
        ...         max_seq_len=10,
        ...         task_labels_dir=task_labels_dir,
        ...         seq_sampling_strategy="to_end",
        ...     )

        If the cohort directory doesn't exist, an error is raised.

        >>> with tempfile.TemporaryDirectory() as tmpdir: # Error as cohort dir doesn't exist
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=Path(tmpdir) / "non_existent",
        ...         max_seq_len=10,
        ...     )
        Traceback (most recent call last):
            ...
        FileNotFoundError: tensorized_cohort_dir must be a valid directory. Got ...

        If the task labels directory doesn't exist, an error is raised.

        >>> with tempfile.TemporaryDirectory() as tmpdir: # Error as task labels dir doesn't exist
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=tmpdir,
        ...         max_seq_len=10,
        ...         task_labels_dir=Path(tmpdir) / "non_existent",
        ...     )
        Traceback (most recent call last):
            ...
        FileNotFoundError: If specified, task_labels_dir must be a valid directory. Got ...

        If the subsequence sampling strategy is not TO_END when a task is specified an error is raised.

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_root = Path(tmpdir) / "tensorized"
        ...     cohort_root.mkdir()
        ...     task_labels_dir = Path(tmpdir) / "task_labels"
        ...     task_labels_dir.mkdir()
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=cohort_root,
        ...         max_seq_len=10,
        ...         task_labels_dir=task_labels_dir,
        ...         seq_sampling_strategy="random",
        ...     )
        Traceback (most recent call last):
            ...
        ValueError: Not sampling data till the end of the sequence when predicting for a specific task is not
        permitted! This is because there is no use-case we know of where you would want to do this. If you
        disagree, please let us know via a GitHub issue.

        If the subsequence sampling strategy or static inclusion mode is not valid, an error is raised.

        >>> MEDSTorchDataConfig(tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="foobar")
        Traceback (most recent call last):
            ...
        ValueError: Invalid subsequence sampling strategy: foobar
        >>> MEDSTorchDataConfig(tensorized_cohort_dir=".", max_seq_len=3, static_inclusion_mode="foobar")
        Traceback (most recent call last):
            ...
        ValueError: Invalid static inclusion mode: foobar

        STEP_THROUGH sampling requires exactly one of ``step_through_stride`` (elements to
        advance between consecutive windows) or ``step_through_overlap`` (elements consecutive
        windows should share). Both are in the same unit as ``max_seq_len`` — events in SEM
        mode, measurements in SM mode. Leaving both unset, or setting both, raises:

        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="step_through"
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of step_through_stride or step_through_overlap must be set when
        seq_sampling_strategy is STEP_THROUGH; got step_through_stride=None,
        step_through_overlap=None.
        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="step_through",
        ...     step_through_stride=2, step_through_overlap=1,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of step_through_stride or step_through_overlap must be set when
        seq_sampling_strategy is STEP_THROUGH; got step_through_stride=2,
        step_through_overlap=1.

        ``step_through_stride`` must be a positive integer. Zero / negative / non-int values
        are rejected, and ``bool`` is explicitly rejected because it is a subclass of ``int``
        in Python (so ``isinstance(True, int)`` would otherwise silently accept it as stride 1):

        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="step_through",
        ...     step_through_stride=0,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: step_through_stride must be a positive integer when seq_sampling_strategy is
        STEP_THROUGH; got 0.
        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="step_through",
        ...     step_through_stride=True,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: step_through_stride must be a positive integer when seq_sampling_strategy is
        STEP_THROUGH; got True.

        ``step_through_overlap`` must be a non-negative integer (``0`` = contiguous
        non-overlapping windows). ``bool`` is again explicitly rejected:

        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="step_through",
        ...     step_through_overlap=-1,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: step_through_overlap must be a non-negative integer when seq_sampling_strategy
        is STEP_THROUGH; got -1.
        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="step_through",
        ...     step_through_overlap=True,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: step_through_overlap must be a non-negative integer when seq_sampling_strategy
        is STEP_THROUGH; got True.

        Conversely, setting ``step_through_stride`` or ``step_through_overlap`` with any other
        strategy is also rejected, because the fields have no effect outside STEP_THROUGH and
        silently accepting them would mask configuration mistakes:

        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="random",
        ...     step_through_stride=2,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: step_through_stride may only be set when seq_sampling_strategy is STEP_THROUGH;
        got strategy random with stride 2.
        >>> MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="random",
        ...     step_through_overlap=0,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: step_through_overlap may only be set when seq_sampling_strategy is STEP_THROUGH;
        got strategy random with overlap 0.
    """

    # MEDS Dataset Information
    tensorized_cohort_dir: str

    # Sequence lengths and padding
    max_seq_len: int
    seq_sampling_strategy: SubsequenceSamplingStrategy = SubsequenceSamplingStrategy.RANDOM
    padding_side: PaddingSide = PaddingSide.RIGHT

    # Static Data
    static_inclusion_mode: StaticInclusionMode = StaticInclusionMode.INCLUDE

    # Task Labels
    task_labels_dir: str | None = None

    # Output Shape & Masking
    batch_mode: BatchMode = BatchMode.SM

    # Extra output
    include_window_last_observed_in_schema: bool = False

    # STEP_THROUGH sampling-specific options. Exactly one of `step_through_stride` or
    # `step_through_overlap` must be set when `seq_sampling_strategy == STEP_THROUGH`; both
    # must be `None` for all other strategies. Both are specified in the same unit as
    # `max_seq_len` — events in SEM mode, measurements in SM mode.
    #
    # - `step_through_stride`: the number of sequence elements to advance between consecutive
    #   windows. Must be a positive integer `<=` the effective window width (otherwise some
    #   elements would be skipped between windows); validated at dataset construction time
    #   because the effective window can vary per subject in SM+PREPEND mode.
    # - `step_through_overlap`: the number of sequence elements consecutive windows should
    #   share. Equivalent to `stride = effective_window - overlap`, but more convenient when
    #   the user wants "no overlap" (`overlap=0`) or "overlap by N" without having to think
    #   about the effective window. Must be a non-negative integer strictly less than the
    #   effective window.
    # - `include_subject_window_counts_in_batch`: when True, the collated `MEDSTorchBatch` will
    #   populate its `n_subject_windows` tensor (shape `[batch_size]`) with the number of
    #   dataset elements each sample's subject expands into, so downstream losses can reweight
    #   by `1 / n_subject_windows` to undo step-through oversampling of long sequences.
    step_through_stride: int | None = None
    step_through_overlap: int | None = None
    include_subject_window_counts_in_batch: bool = False

    @classmethod
    def add_to_config_store(cls, group: str | None = None):
        """Adds this class to the Hydra config store such that instantiation will create it natively.

        Args:
            group: The group name to register this class under.

        Examples:
            >>> MEDSTorchDataConfig.add_to_config_store()
            >>> cs = ConfigStore.instance()
            >>> cs.repo["MEDSTorchDataConfig.yaml"]
            ConfigNode(name='MEDSTorchDataConfig.yaml',
                       node={'tensorized_cohort_dir': '???',
                             'max_seq_len': '???',
                             'seq_sampling_strategy': <SubsequenceSamplingStrategy.RANDOM: 'random'>,
                             'padding_side': <PaddingSide.RIGHT: 'right'>,
                             'static_inclusion_mode': <StaticInclusionMode.INCLUDE: 'include'>,
                             'task_labels_dir': None,
                             'batch_mode': <BatchMode.SM: 'SM'>,
                             'include_window_last_observed_in_schema': False,
                             'step_through_stride': None,
                             'step_through_overlap': None,
                             'include_subject_window_counts_in_batch': False,
                             '_target_': 'meds_torchdata.config.MEDSTorchDataConfig'},
                       group=None,
                       package=None,
                       provider=None)

        With the `_target_` key set to the class name, this allows for instantiation of the class via Hydra:

            >>> from omegaconf import DictConfig
            >>> from hydra import compose, initialize
            >>> with initialize(version_base=None, config_path=".", job_name="test"):
            ...     cfg = compose(
            ...         config_name="MEDSTorchDataConfig.yaml",
            ...         overrides=[f"tensorized_cohort_dir={tensorized_MEDS_dataset!s}", "max_seq_len=10"]
            ...     )
            >>> cfg
            {'tensorized_cohort_dir': '/tmp/tmp...',
             'max_seq_len': 10,
             'seq_sampling_strategy': <SubsequenceSamplingStrategy.RANDOM: 'random'>,
             'padding_side': <PaddingSide.RIGHT: 'right'>,
             'static_inclusion_mode': <StaticInclusionMode.INCLUDE: 'include'>,
             'task_labels_dir': None,
             'batch_mode': <BatchMode.SM: 'SM'>,
             'include_window_last_observed_in_schema': False,
             'step_through_stride': None,
             'step_through_overlap': None,
             'include_subject_window_counts_in_batch': False,
             '_target_': 'meds_torchdata.config.MEDSTorchDataConfig'}
            >>> from hydra.utils import instantiate
            >>> instantiate(cfg)
            MEDSTorchDataConfig(tensorized_cohort_dir=PosixPath('/tmp/tmp...'),
                                max_seq_len=10,
                                seq_sampling_strategy=<SubsequenceSamplingStrategy.RANDOM: 'random'>,
                                padding_side=<PaddingSide.RIGHT: 'right'>,
                                static_inclusion_mode=<StaticInclusionMode.INCLUDE: 'include'>,
                                task_labels_dir=None,
                                batch_mode=<BatchMode.SM: 'SM'>,
                                include_window_last_observed_in_schema=False,
                                step_through_stride=None,
                                step_through_overlap=None,
                                include_subject_window_counts_in_batch=False)

        Note that Hydra's CLI parameters with structured configs recognize that the `StrEnum` classes are
        enums, but fails to recognize that they accept lowercased names as the names of the class members are
        all upper-case. This means that you need to use upper case names for enum variables if you overwrite a
        parameter in the CLI for this config once it is added to the config store.

            >>> with initialize(version_base=None, config_path=".", job_name="test"):
            ...     cfg = compose(
            ...         config_name="MEDSTorchDataConfig.yaml",
            ...         overrides=[
            ...             f"tensorized_cohort_dir={tensorized_MEDS_dataset!s}",
            ...             "max_seq_len=10",
            ...             "seq_sampling_strategy=to_end",
            ...         ]
            ...     )
            Traceback (most recent call last):
                ...
            hydra.errors.ConfigCompositionException: Error merging override seq_sampling_strategy=to_end
            >>> with initialize(version_base=None, config_path=".", job_name="test"):
            ...     cfg = compose(
            ...         config_name="MEDSTorchDataConfig.yaml",
            ...         overrides=[
            ...             f"tensorized_cohort_dir={tensorized_MEDS_dataset!s}",
            ...             "max_seq_len=10",
            ...             "seq_sampling_strategy=TO_END",
            ...         ]
            ...     )
            >>> instantiate(cfg)
            MEDSTorchDataConfig(tensorized_cohort_dir=PosixPath('/tmp/tmp...'),
                                max_seq_len=10,
                                seq_sampling_strategy=<SubsequenceSamplingStrategy.TO_END: 'to_end'>,
                                padding_side=<PaddingSide.RIGHT: 'right'>,
                                static_inclusion_mode=<StaticInclusionMode.INCLUDE: 'include'>,
                                task_labels_dir=None,
                                batch_mode=<BatchMode.SM: 'SM'>,
                                include_window_last_observed_in_schema=False,
                                step_through_stride=None,
                                step_through_overlap=None,
                                include_subject_window_counts_in_batch=False)

        You can also add the config to a group

            >>> MEDSTorchDataConfig.add_to_config_store("my_group/my_subgroup")
            >>> cs = ConfigStore.instance()
            >>> cs.repo["my_group"]["my_subgroup"]["MEDSTorchDataConfig.yaml"]
            ConfigNode(name='MEDSTorchDataConfig.yaml',
                       node={'tensorized_cohort_dir': '???',
                             'max_seq_len': '???',
                             'seq_sampling_strategy': <SubsequenceSamplingStrategy.RANDOM: 'random'>,
                             'padding_side': <PaddingSide.RIGHT: 'right'>,
                             'static_inclusion_mode': <StaticInclusionMode.INCLUDE: 'include'>,
                             'task_labels_dir': None,
                             'batch_mode': <BatchMode.SM: 'SM'>,
                             'include_window_last_observed_in_schema': False,
                             'step_through_stride': None,
                             'step_through_overlap': None,
                             'include_subject_window_counts_in_batch': False,
                             '_target_': 'meds_torchdata.config.MEDSTorchDataConfig'},
                       group='my_group/my_subgroup',
                       package=None,
                       provider=None)
        """

        # 1. Register it
        cs = ConfigStore.instance()
        cs.store(name=cls.__name__, group=group, node=cls)

        # 2. Add the target
        node = cs.repo
        if group is not None:
            for key in group.split("/"):
                node = node[key]

        node = node[f"{cls.__name__}.yaml"].node
        with open_dict(node):
            node["_target_"] = f"{cls.__module__}.{cls.__name__}"

    def __post_init__(self):
        self.tensorized_cohort_dir = Path(self.tensorized_cohort_dir)
        if not self.tensorized_cohort_dir.is_dir():
            raise FileNotFoundError(
                "tensorized_cohort_dir must be a valid directory. "
                f"Got {self.tensorized_cohort_dir.resolve()!s}"
            )

        match self.static_inclusion_mode:
            case str() if self.static_inclusion_mode in {x.value for x in StaticInclusionMode}:
                self.static_inclusion_mode = StaticInclusionMode(self.static_inclusion_mode)
            case StaticInclusionMode():  # pragma: no cover
                pass
            case _:
                raise ValueError(f"Invalid static inclusion mode: {self.static_inclusion_mode}")

        match self.seq_sampling_strategy:
            case str() if self.seq_sampling_strategy in {x.value for x in SubsequenceSamplingStrategy}:
                self.seq_sampling_strategy = SubsequenceSamplingStrategy(self.seq_sampling_strategy)
            case SubsequenceSamplingStrategy():  # pragma: no cover
                pass
            case _:
                raise ValueError(f"Invalid subsequence sampling strategy: {self.seq_sampling_strategy}")

        if self.task_labels_dir is not None:
            self.task_labels_dir = Path(self.task_labels_dir)
            if not self.task_labels_dir.is_dir():
                raise FileNotFoundError(
                    "If specified, task_labels_dir must be a valid directory. "
                    f"Got {self.task_labels_dir.resolve()!s}"
                )
            if self.seq_sampling_strategy != SubsequenceSamplingStrategy.TO_END:
                raise ValueError(
                    "Not sampling data till the end of the sequence when predicting for a specific task is "
                    "not permitted! This is because there is no use-case we know of where you would want to "
                    "do this. If you disagree, please let us know via a GitHub issue."
                )

        if self.seq_sampling_strategy == SubsequenceSamplingStrategy.STEP_THROUGH:
            # Exactly one of `step_through_stride` or `step_through_overlap` must be set.
            n_set = (self.step_through_stride is not None) + (self.step_through_overlap is not None)
            if n_set != 1:
                raise ValueError(
                    "Exactly one of step_through_stride or step_through_overlap must be set when "
                    "seq_sampling_strategy is STEP_THROUGH; got "
                    f"step_through_stride={self.step_through_stride!r}, "
                    f"step_through_overlap={self.step_through_overlap!r}."
                )
            # `bool` is a subclass of `int`, so `isinstance(True, int)` is `True`. Reject
            # `bool` explicitly so `step_through_stride=True` doesn't silently behave like
            # stride 1 (or `step_through_overlap=True` like overlap 1).
            if self.step_through_stride is not None and (
                isinstance(self.step_through_stride, bool)
                or not isinstance(self.step_through_stride, int)
                or self.step_through_stride <= 0
            ):
                raise ValueError(
                    "step_through_stride must be a positive integer when seq_sampling_strategy is "
                    f"STEP_THROUGH; got {self.step_through_stride!r}."
                )
            if self.step_through_overlap is not None and (
                isinstance(self.step_through_overlap, bool)
                or not isinstance(self.step_through_overlap, int)
                or self.step_through_overlap < 0
            ):
                raise ValueError(
                    "step_through_overlap must be a non-negative integer when seq_sampling_strategy "
                    f"is STEP_THROUGH; got {self.step_through_overlap!r}."
                )
        else:
            if self.step_through_stride is not None:
                raise ValueError(
                    "step_through_stride may only be set when seq_sampling_strategy is STEP_THROUGH; "
                    f"got strategy {self.seq_sampling_strategy} with stride {self.step_through_stride!r}."
                )
            if self.step_through_overlap is not None:
                raise ValueError(
                    "step_through_overlap may only be set when seq_sampling_strategy is STEP_THROUGH; "
                    f"got strategy {self.seq_sampling_strategy} with overlap {self.step_through_overlap!r}."
                )

    @property
    def code_metadata_fp(self) -> Path:
        """Return the code metadata file for this cohort.

        The path need not exist to be returned.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     cfg = MEDSTorchDataConfig(Path(tmpdir), max_seq_len=10)
            >>> cfg.code_metadata_fp
            PosixPath('/tmp/tmp.../metadata/codes.parquet')
        """
        return self.tensorized_cohort_dir / "metadata" / "codes.parquet"

    @cached_property
    def vocab_size(self) -> int:
        """Reads the code indices from the metadata file and returns the size of the vocabulary.

        The vocabulary size is the maximum index in the code metadata file plus one. This is a cached property
        to avoid reading the file multiple times.

        Examples:
            >>> df = pl.DataFrame({"code/vocab_index": [0, 1, 3]})
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     tensorized_root = Path(tmpdir)
            ...     metadata_fp = tensorized_root / "metadata" / "codes.parquet"
            ...     metadata_fp.parent.mkdir(parents=True)
            ...     df.write_parquet(metadata_fp)
            ...     cfg = MEDSTorchDataConfig(tensorized_root, max_seq_len=10)
            ...     print(cfg.vocab_size)
            4
        """
        df = pl.read_parquet(self.code_metadata_fp, columns=["code/vocab_index"], use_pyarrow=True)
        return df.select(pl.col("code/vocab_index")).max().item() + 1

    @property
    def schema_dir(self) -> Path:
        """Return the schema directory for the tensorized cohort.

        The path need not exist to be returned.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     cfg = MEDSTorchDataConfig(Path(tmpdir), max_seq_len=10)
            >>> cfg.schema_dir
            PosixPath('/tmp/tmp.../tokenization/schemas')
        """
        return self.tensorized_cohort_dir / "tokenization" / "schemas"

    @property
    def schema_fps(self) -> Generator[tuple[str, Path], None, None]:
        """Yield shard names and schema paths for existent schema files.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     tensorized_root = Path(tmpdir)
            ...     schema_dir = tensorized_root / "tokenization" / "schemas"
            ...     schema_dir.mkdir(parents=True)
            ...     (schema_dir / "shard_A.parquet").touch()
            ...     (schema_dir / "shard_B.json").touch()
            ...     (schema_dir / "shard_C/").mkdir()
            ...     (schema_dir / "shard_C" / "0.parquet").touch()
            ...     (schema_dir / "shard_C" / "1.parquet").touch()
            ...     (schema_dir / "shard_D/").mkdir()
            ...     cfg = MEDSTorchDataConfig(tensorized_root, max_seq_len=10)
            ...     for shard, fp in cfg.schema_fps:
            ...         print(shard, str(fp.relative_to(tensorized_root)))
            shard_A tokenization/schemas/shard_A.parquet
            shard_C/0 tokenization/schemas/shard_C/0.parquet
            shard_C/1 tokenization/schemas/shard_C/1.parquet
        """

        for schema_fp in sorted(self.schema_dir.rglob("*.parquet")):
            shard = str(schema_fp.relative_to(self.schema_dir).with_suffix(""))
            yield shard, schema_fp

    @property
    def task_labels_fps(self) -> list[Path] | None:
        """Returns the list of task label files for this configuration, or `None` if no task is specified.

        Returned files must exist; if no such files exist, will return an empty list.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     tensorized_root = Path(tmpdir) / "tensorized"
            ...     tensorized_root.mkdir()
            ...     cfg_no_task = MEDSTorchDataConfig(tensorized_root, 2)
            ...     print(f"No task dir: {cfg_no_task.task_labels_fps}")
            ...     task_labels_dir = Path(tmpdir) / "task_labels"
            ...     task_labels_dir.mkdir()
            ...     (task_labels_dir / "labels_1.parquet").touch()
            ...     (task_labels_dir / "nested").mkdir()
            ...     (task_labels_dir / "nested/labels_2.parquet").touch()
            ...     cfg_task = MEDSTorchDataConfig(
            ...         tensorized_root, 2, task_labels_dir=task_labels_dir, seq_sampling_strategy="to_end"
            ...     )
            ...     print(f"Task dir: {cfg_task.task_labels_fps}")
            No task dir: None
            Task dir: [PosixPath('/tmp/.../task_labels/labels_1.parquet'),
                       PosixPath('/tmp/.../task_labels/nested/labels_2.parquet')]
        """

        return sorted(self.task_labels_dir.rglob("*.parquet")) if self.task_labels_dir else None

    def process_dynamic_data(
        self,
        data: JointNestedRaggedTensorDict,
        n_static_seq_els: int | None = None,
        rng: np.random.Generator | int | None = None,
        explicit_end: int | None = None,
    ) -> JointNestedRaggedTensorDict:
        """This processes the dynamic data for a subject, including subsampling and flattening.

        Args:
            data: The dynamic data for the subject.
            n_static_seq_els: The number of static measurements for the given patient. This is only used
                if the static inclusion mode is `StaticInclusionMode.PREPEND`, in which case it must not be
                `None`.
            rng: The random seed to use for subsequence sampling. If `None`, the default rng is used. If an
                integer, a new rng is created with that seed.
            explicit_end: An optional measurement-level end index for the window. When set,
                `process_dynamic_data` returns
                `data[max(0, explicit_end - effective_max_seq_len) : explicit_end]` after the
                mode-appropriate flatten, bypassing the sampler entirely. This is the
                `STEP_THROUGH`+`BatchMode.SM` path: the dataset's index expansion pre-computes
                per-window measurement ends that can terminate mid-event, and passes each
                one through here so the window is measurement-level precise regardless of how
                many measurements an event has. `None` for every other caller — including
                `STEP_THROUGH` in SEM mode, which goes through the normal
                `STEP_THROUGH → TO_END` sampler path.

        Returns:
            The processed dynamic data, still in a `JointNestedRaggedTensorDict` format.

        Examples:
            >>> from nested_ragged_tensors.ragged_numpy import pprint_dense
            >>> data = JointNestedRaggedTensorDict({
            ...     "time_delta": [1, 2, 3, 4, 5, 6, 7],
            ...     "code": [[10, 11], [20, 21], [30], [40], [50, 51, 52], [60], [70, 71, 72, 73]],
            ... })

            If the config says to sample until the end, we'll just grab the last three elements.

            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=3, seq_sampling_strategy="to_end", batch_mode="SEM"
            ... )
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            time_delta
            [5 6 7]
            .
            ---
            .
            dim1/mask
            [[ True  True  True False]
             [ True False False False]
             [ True  True  True  True]]
            .
            code
            [[50 51 52  0]
             [60  0  0  0]
             [70 71 72 73]]

            We can also pass the number of sequence elements that should be reserved for static sequence
            elements to functionally reduce the effective max sequence length we select among the dynamic
            data. This is only used in `StaticInclusionMode.PREPEND` mode, and is ignored otherwise (without
            an error being raised!). Note that the reserved sequence element only affects the first
            (sequential) dimension of the nested ragged tensor.

            >>> pprint_dense(cfg.process_dynamic_data(data, n_static_seq_els=1).to_dense())
            time_delta
            [5 6 7]
            .
            ---
            .
            dim1/mask
            [[ True  True  True False]
             [ True False False False]
             [ True  True  True  True]]
            .
            code
            [[50 51 52  0]
             [60  0  0  0]
             [70 71 72 73]]
            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=3, seq_sampling_strategy="to_end", batch_mode="SEM",
            ...     static_inclusion_mode="prepend"
            ... )
            >>> pprint_dense(cfg.process_dynamic_data(data, n_static_seq_els=1).to_dense())
            time_delta
            [6 7]
            .
            ---
            .
            dim1/mask
            [[ True False False False]
             [ True  True  True  True]]
            .
            code
            [[60  0  0  0]
             [70 71 72 73]]

            If we flatten the tensors, then we get only 1D tensors for both, and the time elements that are
            added to account for the longer length are imputed to zero. Note we've increased the `max_seq_len`
            to 5 to show some non-imputed time-deltas.

            >>> cfg = MEDSTorchDataConfig(".", max_seq_len=5, seq_sampling_strategy="to_end")
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            code
            [60 70 71 72 73]
            .
            time_delta
            [6 7 0 0 0]
            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=5, seq_sampling_strategy="to_end",
            ...     static_inclusion_mode="prepend"
            ... )
            >>> pprint_dense(cfg.process_dynamic_data(data, n_static_seq_els=3).to_dense())
            code
            [72 73]
            .
            time_delta
            [0 0]

            If we sample from the start, we'll just grab the first three elements.

            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=3, seq_sampling_strategy="from_start", batch_mode="SEM"
            ... )
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            time_delta
            [1 2 3]
            .
            ---
            .
            dim1/mask
            [[ True  True]
             [ True  True]
             [ True False]]
            .
            code
            [[10 11]
             [20 21]
             [30  0]]

            Again, if we flatten the tensors, we get only 1D tensors for both.

            >>> cfg = MEDSTorchDataConfig(".", max_seq_len=3, seq_sampling_strategy="from_start")
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            code
            [10 11 20]
            .
            time_delta
            [1 0 2]

            Random sampling is non-deterministic, but can be fixed by a seed.

            >>> cfg = MEDSTorchDataConfig(".", max_seq_len=3, seq_sampling_strategy="random")
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=1).to_dense())
            code
            [40 50 51]
            .
            time_delta
            [4 5 0]
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=1).to_dense())
            code
            [40 50 51]
            .
            time_delta
            [4 5 0]
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=3).to_dense())
            code
            [60 70 71]
            .
            time_delta
            [6 7 0]

            `balanced_random` lets the sliding window overhang the left or right edge of the
            sequence, giving every event a uniform `max_seq_len / (seq_len + max_seq_len - 1)`
            chance of being included. When the window overhangs a boundary, the returned slice
            is *shorter* than `max_seq_len` — the collator pads to the longest element in the
            batch downstream. Here `seq_len` is 14 (in SM mode the measurement tensor is
            flattened first), `max_seq_len` is 3, so the start offset is drawn uniformly from
            `{-2, -1, ..., 13}`.

            >>> cfg = MEDSTorchDataConfig(".", max_seq_len=3, seq_sampling_strategy="balanced_random")
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=23).to_dense())
            code
            [10]
            .
            time_delta
            [1]
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=30).to_dense())
            code
            [10 11]
            .
            time_delta
            [1 0]
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=7).to_dense())
            code
            [73]
            .
            time_delta
            [0]
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=9).to_dense())
            code
            [30 40 50]
            .
            time_delta
            [3 4 5]

        If we pass in an invalid number of static sequence elements to reserve, we get an error.

            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=3, seq_sampling_strategy="random", static_inclusion_mode="prepend"
            ... )
            >>> cfg.process_dynamic_data(data, n_static_seq_els=0)
            Traceback (most recent call last):
                ...
            ValueError: When self.static_inclusion_mode=prepend, n_static_seq_els must be a positive integer.
                Got 0
        """

        if self.batch_mode == BatchMode.SM:
            data = data.flatten()

        seq_len = len(data)
        max_seq_len = self.max_seq_len

        if self.static_inclusion_mode == StaticInclusionMode.PREPEND:
            if not isinstance(n_static_seq_els, int) or n_static_seq_els <= 0:
                raise ValueError(
                    f"When self.static_inclusion_mode={self.static_inclusion_mode}, "
                    f"n_static_seq_els must be a positive integer. Got {n_static_seq_els}"
                )

            max_seq_len -= n_static_seq_els

        # `explicit_end` (set by `STEP_THROUGH` in `BatchMode.SM`) semantically means "don't
        # go past this measurement". We can honor it by telling the sampler to act as if the
        # sequence were truncated there — the `STEP_THROUGH → TO_END` delegation in
        # `subsample_st_offset` then naturally returns `explicit_end - max_seq_len`, so the
        # resulting window is `[explicit_end - max_seq_len, explicit_end)` without any
        # sampler-bypass branch. `min(seq_len, ...)` guards against an over-ambitious caller.
        effective_seq_len = min(seq_len, explicit_end) if explicit_end is not None else seq_len

        st = self.seq_sampling_strategy.subsample_st_offset(effective_seq_len, max_seq_len, rng=rng)
        end = st + max_seq_len

        # Clamp the resulting slice: `BALANCED_RANDOM` can return a negative `st` so the
        # window overhangs the left boundary (yielding a uniform per-event inclusion
        # distribution — padding is handled by the collator downstream); `end` likewise may
        # overhang the right boundary or exceed `seq_len` for short sequences.
        st = max(0, st)
        end = min(seq_len, end)
        return data[st:end]
