"""Contains configuration objects for building a PyTorch dataset from a MEDS dataset.

This module contains configuration objects for building a PyTorch dataset from a MEDS dataset. These include
enumeration objects for categorical options and a general DataClass configuration object for dataset options.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from numpy.random import Generator, default_rng


def resolve_rng(rng: Generator | int | None) -> Generator:
    """Resolve a random number generator from a seed or generator.

    Args:
        rng: Random number generator for random sampling. If None, a new generator is created. If an
            integer, a new generator is created with that seed.

    Returns:
        A random number generator.
    """

    match rng:
        case None:
            return default_rng()
        case int():
            return default_rng(rng)
        case Generator():
            return rng
        case _:
            raise ValueError(f"Invalid random number generator: {rng}!")


class SubsequenceSamplingStrategy(StrEnum):
    """An enumeration of the possible subsequence sampling strategies for the dataset.

    Attributes:
        RANDOM: Randomly sample a subsequence from the full sequence.
        TO_END: Sample a subsequence from the end of the full sequence.
            Note this starts at the last element and moves back.
        FROM_START: Sample a subsequence from the start of the full sequence.

    Methods:
        subsample_st_offset: Subsample starting offset based on maximum sequence length and sampling strategy.
            This method can be used on instances
            (e.g., SubsequenceSamplingStrategy.RANDOM.subsample_st_offset) but is most often used as a static
            class level method for maximal clarity.
    """

    RANDOM = "random"
    TO_END = "to_end"
    FROM_START = "from_start"

    def subsample_st_offset(
        strategy,
        seq_len: int,
        max_seq_len: int,
        rng: Generator | int | None = None,
    ) -> int | None:
        """Subsample starting offset based on maximum sequence length and sampling strategy.

        Args:
            strategy: Strategy for selecting subsequence (RANDOM, TO_END, FROM_START)
            seq_len: Length of the sequence
            max_seq_len: Maximum allowed sequence length
            rng: Random number generator for random sampling. If None, a new generator is created. If an
                integer, a new generator is created with that seed.

        Returns:
            The (integral) start offset within the sequence based on the sampling strategy, or `None` if no
            subsampling is required.

        Examples:
            >>> SubsequenceSamplingStrategy.subsample_st_offset("from_start", 10, 5)
            0
            >>> SubsequenceSamplingStrategy.subsample_st_offset(SubsequenceSamplingStrategy.TO_END, 10, 5)
            5
            >>> SubsequenceSamplingStrategy.subsample_st_offset("random", 10, 5, rng=default_rng(1))
            2
            >>> SubsequenceSamplingStrategy.subsample_st_offset("random", 10, 5, rng=1)
            2
            >>> SubsequenceSamplingStrategy.RANDOM.subsample_st_offset(10, 10) is None
            True
            >>> SubsequenceSamplingStrategy.subsample_st_offset("foo", 10, 5)
            Traceback (most recent call last):
                ...
            ValueError: Invalid subsequence sampling strategy foo!
        """

        if seq_len <= max_seq_len:
            return None

        match strategy:
            case SubsequenceSamplingStrategy.RANDOM:
                return resolve_rng(rng).choice(seq_len - max_seq_len)
            case SubsequenceSamplingStrategy.TO_END:
                return seq_len - max_seq_len
            case SubsequenceSamplingStrategy.FROM_START:
                return 0
            case _:
                raise ValueError(f"Invalid subsequence sampling strategy {strategy}!")


class StaticInclusionMode(StrEnum):
    """An enumeration of the possible vehicles to include static measurements.

    Attributes:
        INCLUDE: Include the static measurements as a separate output key in each batch.
        OMIT: Omit the static measurements entirely.
    """

    INCLUDE = "include"
    OMIT = "omit"


@dataclass
class MEDSTorchDataConfig:
    """A data class for storing configuration options for building a PyTorch dataset from a MEDS dataset.

    Attributes:
        seq_len: The maximum length of sequences to yield from the dataset.
        subseq_sampling_strategy: The subsequence sampling strategy for the dataset.
        subseq_len: The length of the subsequences in the dataset.

    Raises:
        FileNotFoundError: If the task_labels_dir or the tensorized_cohort_dir is not a valid directory.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir: # No error
        ...     cfg = MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=tmpdir,
        ...         max_seq_len=10,
        ...     )
        >>> with tempfile.TemporaryDirectory() as tmpdir: # Error as cohort dir doesn't exist
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=Path(tmpdir) / "non_existent",
        ...         max_seq_len=10,
        ...     )
        Traceback (most recent call last):
            ...
        FileNotFoundError: tensorized_cohort_dir must be a valid directory. Got ...
        >>> with tempfile.TemporaryDirectory() as tmpdir: # Error as task labels dir doesn't exist
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=tmpdir,
        ...         max_seq_len=10,
        ...         task_labels_dir=Path(tmpdir) / "non_existent",
        ...     )
        Traceback (most recent call last):
            ...
        FileNotFoundError: If specified, task_labels_dir must be a valid directory. Got ...
        >>> MEDSTorchDataConfig(tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="foobar")
        Traceback (most recent call last):
            ...
        ValueError: Invalid subsequence sampling strategy: foobar
        >>> MEDSTorchDataConfig(tensorized_cohort_dir=".", max_seq_len=3, static_inclusion_mode="foobar")
        Traceback (most recent call last):
            ...
        ValueError: Invalid static inclusion mode: foobar
    """

    # MEDS Dataset Information
    tensorized_cohort_dir: str

    # Sequence lengths and padding
    max_seq_len: int
    seq_sampling_strategy: SubsequenceSamplingStrategy = SubsequenceSamplingStrategy.RANDOM

    # Static Data
    static_inclusion_mode: StaticInclusionMode = StaticInclusionMode.INCLUDE

    # Task Labels
    task_labels_dir: str | None = None

    # Output Shape & Masking
    do_flatten_tensors: bool = True

    def __post_init__(self):
        self.tensorized_cohort_dir = Path(self.tensorized_cohort_dir)
        if not self.tensorized_cohort_dir.is_dir():
            raise FileNotFoundError(
                "tensorized_cohort_dir must be a valid directory. "
                f"Got {str(self.tensorized_cohort_dir.resolve())}"
            )

        if self.task_labels_dir:
            self.task_labels_dir = Path(self.task_labels_dir)
            if not self.task_labels_dir.is_dir():
                raise FileNotFoundError(
                    "If specified, task_labels_dir must be a valid directory. "
                    f"Got {str(self.task_labels_dir.resolve())}"
                )

        match self.static_inclusion_mode:
            case str() if self.static_inclusion_mode in {x.value for x in StaticInclusionMode}:
                self.static_inclusion_mode = StaticInclusionMode(self.static_inclusion_mode)
            case StaticInclusionMode():
                pass
            case _:
                raise ValueError(f"Invalid static inclusion mode: {self.static_inclusion_mode}")

        match self.seq_sampling_strategy:
            case str() if self.seq_sampling_strategy in {x.value for x in SubsequenceSamplingStrategy}:
                self.seq_sampling_strategy = SubsequenceSamplingStrategy(self.seq_sampling_strategy)
            case SubsequenceSamplingStrategy():
                pass
            case _:
                raise ValueError(f"Invalid subsequence sampling strategy: {self.seq_sampling_strategy}")

    def process_dynamic_data(
        self,
        data: JointNestedRaggedTensorDict,
        rng: Generator | int | None = None,
    ) -> JointNestedRaggedTensorDict:
        """This processes the dynamic data for a subject, including subsampling and flattening.

        Args:
            data: The dynamic data for the subject.
            rng: The random seed to use for subsequence sampling. If `None`, the default rng is used. If an
                integer, a new rng is created with that seed.

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
            ...     ".", max_seq_len=3, seq_sampling_strategy="to_end", do_flatten_tensors=False
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

            If we sample from the start, we'll just grab the first three elements.

            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=3, seq_sampling_strategy="from_start", do_flatten_tensors=False
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
            [52 60 70]
            .
            time_delta
            [0 6 7]
        """

        if self.do_flatten_tensors:
            data = data.flatten()

        seq_len = len(data)
        st = self.seq_sampling_strategy.subsample_st_offset(seq_len, self.max_seq_len, rng=rng)

        if st is None:
            st = 0

        end = min(seq_len, st + self.max_seq_len)
        return data[st:end]
