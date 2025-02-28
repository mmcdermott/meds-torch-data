"""Contains configuration objects for building a PyTorch dataset from a MEDS dataset.

This module contains configuration objects for building a PyTorch dataset from a MEDS dataset. These include
enumeration objects for categorical options and a general DataClass configuration object for dataset options.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from numpy.random import Generator, default_rng


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
        rng: Generator | None = None,
    ) -> int | None:
        """Subsample starting offset based on maximum sequence length and sampling strategy.

        Args:
            strategy: Strategy for selecting subsequence (RANDOM, TO_END, FROM_START)
            seq_len: Length of the sequence
            max_seq_len: Maximum allowed sequence length
            rng: Random number generator for random sampling. If None, a new generator is created.

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
                if rng is None:
                    rng = default_rng()
                return rng.choice(seq_len - max_seq_len)
            case SubsequenceSamplingStrategy.TO_END:
                return seq_len - max_seq_len
            case SubsequenceSamplingStrategy.FROM_START:
                return 0
            case _:
                raise ValueError(f"Invalid subsequence sampling strategy {strategy}!")


class SeqPaddingSide(StrEnum):
    """An enumeration of the possible sequence padding sides for the dataset.

    Attributes:
        LEFT: Pad the sequences on the left side (e.g., [[0, 0, 1, 2, 3], [1, 2, 3, 4, 5]]).
        RIGHT: Pad the sequences on the right side (e.g., [[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]]).
    """

    LEFT = "left"
    RIGHT = "right"


class StaticInclusionMode(StrEnum):
    """An enumeration of the possible vehicles to include static measurements.

    Attributes:
        INCLUDE: Include the static measurements as a separate output key in each batch.
        OMIT: Omit the static measurements entirely.
        PREPEND: Prepend the static measurements to the sequence of time-dependent measurements.
    """

    INCLUDE = "include"
    OMIT = "omit"
    PREPEND = "prepend"


@dataclass
class MEDSTorchDataConfig:
    """A data class for storing configuration options for building a PyTorch dataset from a MEDS dataset.

    Attributes:
        seq_len: The maximum length of sequences to yield from the dataset.
        seq_padding_side: The side to pad the sequences on.
        subseq_sampling_strategy: The subsequence sampling strategy for the dataset.
        subseq_len: The length of the subsequences in the dataset.
    """

    # MEDS Dataset Information
    tensorized_cohort_dir: str

    # Sequence lengths and padding
    max_seq_len: int
    seq_padding_side: SeqPaddingSide = SeqPaddingSide.LEFT
    seq_sampling_strategy: SubsequenceSamplingStrategy = SubsequenceSamplingStrategy.RANDOM

    # Static Data
    static_inclusion_mode: StaticInclusionMode = StaticInclusionMode.INCLUDE

    # Task Labels
    task_labels_dir: str | None = None

    # Output Shape & Masking
    do_flatten_tensors: bool = True
    do_include_event_mask: bool = True

    # Include Metadata in batches
    do_include_subject_id: bool = False
    do_include_subsequence_indices: bool = False
    do_include_start_time: bool = False
    do_include_end_time: bool = False
    do_include_prediction_time: bool = False

    def __post_init__(self):
        self.tensorized_cohort_dir = Path(self.tensorized_cohort_dir)

        if self.task_labels_dir:
            self.task_labels_dir = Path(self.task_labels_dir)
            if not self.task_labels_dir.is_dir():
                raise FileNotFoundError(
                    "If specified, task_labels_dir must be a valid directory. "
                    f"Got {str(self.task_labels_dir.resolve())}"
                )


class MEDSTorchBatch(TypedDict):
    """A type hint for a batch of data from a MEDS dataset.

    A dictionary containing the following keys:
        ...
    """

    code: torch.LongTensor
    mask: torch.BoolTensor
    numeric_value: torch.FloatTensor
    numeric_value_mask: torch.BoolTensor
    time_delta: torch.FloatTensor

    static_code: NotRequired[torch.LongTensor]
    static_numeric_value: NotRequired[torch.FloatTensor]
    static_numeric_value_mask: NotRequired[torch.BoolTensor]
    event_mask: NotRequired[torch.BoolTensor]

    subject_id: NotRequired[torch.LongTensor]
    start_event_index: NotRequired[torch.LongTensor]
    end_event_index: NotRequired[torch.LongTensor]
