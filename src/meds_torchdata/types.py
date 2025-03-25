"""Exports simple type definitions used in MEDS torchdata"""

from collections.abc import Generator
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import ClassVar, NamedTuple, get_args

import torch

from .utils import SEED_OR_RNG, resolve_rng


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
        rng: SEED_OR_RNG = None,
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


class BatchMode(StrEnum):
    """An enumeration of the possible batch modes for the dataset.

    Attributes:
        SEM: Subject-Event-Measurement mode. In this mode, data are represented as 3D tensors of sequences of
             measurements per event per subject, with tensor shapes
             `[batch_size, max_events_per_subject, max_measurements_per_event]`.
        SM: Subject-Measurement mode. In this mode, data are represented as 2D tensors of sequences of
            measurements per subject, without explicit separation between measurements of different events,
            with tensor shapes `[batch_size, max_measurements_per_subject]`.
    """

    SEM = "SEM"
    SM = "SM"


class StaticData(NamedTuple):
    """Simple data structure to hold static data, capturing both codes and numeric values.

    As a `NamedTuple`, can be accessed both by index (e.g. `data[0]`) and by attribute (e.g. `data.code`).

    Attributes:
        code: List of integer codes.
        numeric_value: List of float or None numeric values.
    """

    code: list[int]
    numeric_value: list[float | None]


@dataclass
class MEDSTorchBatch:

    """Simple data structure to hold a batch of MEDS data.

    Can be accessed by attribute (e.g., `batch.code`) or string key (e.g. `batch["code"]`). The elements in
    this tensor can take on several shapes, and keys can be present or omitted, depending on details of
    dataset configuration. To clarify these shape options, we'll define the following terms. Most of these
    terms will also be realized as properties defined on this class for accessing shape variables over the
    batch for convenience.

      - `batch_size` is the number of subjects in the batch.
      - `max_events_per_subject` is the maximum number of events (unique time-points) for any subject in the
        batch.
      - `max_measurements_per_event` is the maximum number of measurements (observed code/value pairs) for any
        event in the batch (across all subjects).
      - `max_measurements_per_subject` is the maximum number of measurements observed across _all_ events for
        any given subject, in total, in the batch.
      - `max_static_measurements_per_subject` is the maximum number of static measurements observed across all
        subjects in the batch.

    There are a few shape "modes" that this batch can be in, depending on the configuration of the source
    dataset. These include:

      - `"SEM"`: In Subject-Event-Measurement (SEM) mode, the data is represented as a tensor of measurements
        per-event, per-subject, with missing values padded in all dimensions.
      - `"SM"`: In Subject-Measurement (SM) mode, the data is represented as a tensor of measurements
        per-subject, with events concatenated in order with neither per-event padding nor explicit separator
        tokens.

    Under each of these modes, different sets of the core attributes take on different consistent shapes.

    Under all modes:

      - Static data elements (`static_code`, `static_numeric_value`, and `static_numeric_value_mask`) are
        of shape `[batch_size, max_static_measurements_per_subject]`.
      - The label tensor, `boolean_value` tensor is of shape `[batch_size]`.

    In SEM Mode:

      - Per-event data (`time_delta_days` & `event_mask`) are of shape `[batch_size, max_events_per_subject]`.
        `time_delta_days` will have no zeros at any position save the last event per subject, for which
        position the time delta to the next event may be unknown.
      - Per-measurement data (`code`, `numeric_value`, & `numeric_value_mask`) are of shape
        `[batch_size, max_events_per_subject, max_measurements_per_event]`.

    In SM Mode:

      - `time_delta_days` is of shape `[batch_size, max_measurements_per_subject]`. Will have zeros at
        measurement indices that do not correspond to the last measurement in an event, or at the last
        measurement in the sequence if the next time-delta is unknown.
      - `event_mask` is omitted.
      - Per-measurement data (`code`, `numeric_value`, & `numeric_value_mask`) are of shape
        `[batch_size, max_measurements_per_subject]`.


    Attributes:
        time_delta_days: Tensor of time deltas between sequence elements, in days.
        event_mask: Boolean tensor indicating whether a given event is present or not.
        code: Measurement code integral vocabulary indices. Equals `PAD_INDEX` when measurements are missing.
        numeric_value: Measurement numeric values. No guaranteed value for padding or missing numeric values.
        numeric_value_mask: Boolean mask indicating whether a given measurement has a numeric value. Values of
            this mask for padding measurements are undefined.
        static_code: Static measurement code integral vocabulary indices. Equals `PAD_INDEX` when measurements
            are missing.
        static_numeric_value: Static measurement numeric values. No guaranteed value for padding or missing
            numeric values.
        static_numeric_value_mask: Boolean mask indicating whether a given static measurement has a numeric
            value.
        boolean_value: Per-sample boolean labels.

    Examples:

    The batch is effectively merely an ordered (by the definition in the class, not order of
    specification), frozen dictionary of tensors, and can be accessed as such:

        >>> batch = MEDSTorchBatch(
        ...     time_delta_days=torch.tensor([[1.0, 2.1], [4.0, 0.2]]),
        ...     event_mask=torch.tensor([[True, True], [True, False]]),
        ...     code=torch.tensor([[[1, 2, 3], [3, 0, 0]], [[5, 6, 0], [0, 0, 0]]]),
        ...     numeric_value=torch.tensor(
        ...         [[[1.0, 0.0, -3.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ...     ),
        ...     numeric_value_mask=torch.tensor([
        ...         [[True, False, True], [False, False, False]],
        ...         [[False, True, False], [True, True, True]] # Note the padding values may be  True or False
        ...     ]),
        ... )
        >>> print(batch["code"])
        tensor([[[1, 2, 3],
                 [3, 0, 0]],
        <BLANKLINE>
                [[5, 6, 0],
                 [0, 0, 0]]])
        >>> print(batch["event_mask"])
        tensor([[ True,  True],
                [ True, False]])
        >>> print(list(batch.keys()))
        ['code', 'numeric_value', 'numeric_value_mask', 'time_delta_days', 'event_mask']
        >>> print(list(batch.values()))
        [tensor(...), tensor(...), tensor(...), tensor(...), tensor(...)]
        >>> print(list(batch.items()))
        [('code', tensor(...)), ('numeric_value', tensor(...)), ('numeric_value_mask', tensor(...)),
         ('time_delta_days', tensor(...)), ('event_mask', tensor(...)]
        >>> batch["code"] = torch.tensor([[[1, 2, 3], [3, 0, 0]], [[5, 6, 0], [0, 0, 0]]])
        Traceback (most recent call last):
            ...
        ValueError: MEDSTorchBatch is immutable!

    Though note that if you manually define something in a batch to be `None`, it will not be present in
    the keys/values/items:

        >>> batch = MEDSTorchBatch(
        ...     time_delta_days=torch.tensor([[1.0, 2.1], [4.0, 0.2]]),
        ...     event_mask=torch.tensor([[True, True], [True, False]]),
        ...     code=torch.tensor([[[1, 2, 3], [3, 0, 0]], [[5, 6, 0], [0, 0, 0]]]),
        ...     numeric_value=torch.tensor(
        ...         [[[1.0, 0.0, -3.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ...     ),
        ...     numeric_value_mask=torch.tensor([
        ...         [[True, False, True], [False, False, False]],
        ...         [[False, True, False], [True, True, True]]
        ...     ]),
        ...     boolean_value=None,
        ... )
        >>> print(list(batch.keys()))
        ['code', 'numeric_value', 'numeric_value_mask', 'time_delta_days', 'event_mask']

    The batch can also be accessed by attribute, and has default values for allowed fields:

        >>> print(batch.event_mask)
        tensor([[ True,  True],
                [ True, False]])
        >>> print(batch.boolean_value)
        None

    The batch has a number of properties that can be accessed for convenience:

        >>> print(batch.mode)
        SEM
        >>> print(batch.has_static)
        False
        >>> print(batch.has_labels)
        False
        >>> print(batch.batch_size)
        2
        >>> print(batch.max_events_per_subject)
        2
        >>> print(batch.max_measurements_per_event)
        3
        >>> print(batch.max_measurements_per_subject)
        None
        >>> print(batch.max_static_measurements_per_subject)
        None

    The batch can also be constructed with static data and labels, and in SM mode instead of SEM mode:

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[1, 2, 3, 3], [5, 6, 0, 0]]),
        ...     numeric_value=torch.tensor([[1.0, 0.0, -3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        ...     numeric_value_mask=torch.tensor([[True, False, True, False], [False, True, False, True]]),
        ...     time_delta_days=torch.tensor([[1.0, 0.0, 0.0, 2.0], [4.0, 0.0, 0.0, 0.0]]),
        ...     static_code=torch.tensor([[1], [5]]),
        ...     static_numeric_value=torch.tensor([[1.0], [0.0]]),
        ...     static_numeric_value_mask=torch.tensor([[True], [True]]),
        ...     boolean_value=torch.tensor([True, False]),
        ... )
        >>> print(batch.mode)
        SM
        >>> print(batch.has_static)
        True
        >>> print(batch.has_labels)
        True
        >>> print(batch.batch_size)
        2
        >>> print(batch.max_events_per_subject)
        None
        >>> print(batch.max_measurements_per_event)
        None
        >>> print(batch.max_measurements_per_subject)
        4
        >>> print(batch.max_static_measurements_per_subject)
        1
        >>> print(batch.time_delta_days)
        tensor([[1., 0., 0., 2.],
                [4., 0., 0., 0.]])
        >>> print(batch["event_mask"])
        None
        >>> print(batch["boolean_value"])
        tensor([ True, False])

    The batch will automatically validate tensor shapes, types, and presence vs. omission. In particular,
    the code, numeric_value, numeric_value_mask, and time_delta_days tensors are required, and must be in
    their correct types:

        >>> batch = MEDSTorchBatch()
        Traceback (most recent call last):
            ...
        ValueError: Required tensor code is missing!
        >>> batch = MEDSTorchBatch(code="foobar")
        Traceback (most recent call last):
            ...
        TypeError: Field 'code' expected type <class 'torch.LongTensor'>, got type <class 'str'>.
        >>> batch = MEDSTorchBatch(code=torch.tensor([1.]))
        Traceback (most recent call last):
            ...
        TypeError: Field 'code' expected type <class 'torch.LongTensor'>, got type <class 'torch.Tensor'>.
        >>> batch = MEDSTorchBatch(code=torch.tensor([1]))
        Traceback (most recent call last):
            ...
        ValueError: Required tensor numeric_value is missing!
        >>> batch = MEDSTorchBatch(code=torch.tensor([1]), numeric_value=torch.tensor([1.]))
        Traceback (most recent call last):
            ...
        ValueError: Required tensor numeric_value_mask is missing!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([1]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Required tensor time_delta_days is missing!

    In addition, the shapes of the tensors must be consistent. To begin with, the code tensor's shape must
    correctly align with one of the allowed modes (SEM or SM):

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([1]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([1.]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Code shape must have length either 2 (SM mode) or 3 (SEM mode); got shape torch.Size([1])!

    If the code shape is in SM mode, the remaining tensors must have the correct shapes for that mode:

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[1]]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([1.]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 1) for time_delta_days, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[1]]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([[1.]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 1) for numeric_value, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[1]]),
        ...     numeric_value=torch.tensor([[1.]]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([[1.]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 1) for numeric_value_mask, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[1]]),
        ...     numeric_value=torch.tensor([[1.]]),
        ...     numeric_value_mask=torch.tensor([[True]]),
        ...     time_delta_days=torch.tensor([[1.]]),
        ... )

    You also can't provide an event mask in SM mode:

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[1]]),
        ...     numeric_value=torch.tensor([[1.]]),
        ...     numeric_value_mask=torch.tensor([[True]]),
        ...     time_delta_days=torch.tensor([[1.]]),
        ...     event_mask=torch.tensor([[True]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Event mask should not be provided in SM mode!

    If the code shape is in SEM mode, the remaining tensors must similarly have the correct shapes for
    that mode, and you _must_ provide an event mask:

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([1.]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Event mask must be provided in SEM mode!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([1.]),
        ...     event_mask=torch.tensor([True]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 2) for time_delta_days, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([True]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 2) for event_mask, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([1.]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 2, 2) for numeric_value, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([True]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 2, 2) for numeric_value_mask, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ... )

    If you provide static data, you must provide both the static code and numeric value tensors:

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     static_code=torch.tensor([1, 2]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Static numeric value and mask must both be provided if static codes are!

    You can't provide static numeric values without static codes:

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     static_numeric_value=torch.tensor([1., 2.]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Static numeric value and mask should not be provided without codes!

    Static data tensors must also be provided with consistent shapes, both internally and with respect to
    the other tensors in that the batch size must be conserved.

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     static_code=torch.tensor([1, 2]),
        ...     static_numeric_value=torch.tensor([1.]),
        ...     static_numeric_value_mask=torch.tensor([True, False, True]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected 2D static data tensors with a matching batch size (1), but got static_code shape
                    torch.Size([2])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     static_code=torch.tensor([[1, 2]]),
        ...     static_numeric_value=torch.tensor([1.]),
        ...     static_numeric_value_mask=torch.tensor([True, False, True]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 2) for static_numeric_value, but got torch.Size([1])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     static_code=torch.tensor([[1, 2]]),
        ...     static_numeric_value=torch.tensor([[1., 0.]]),
        ...     static_numeric_value_mask=torch.tensor([True, False, True]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1, 2) for static_numeric_value_mask, but got torch.Size([3])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     static_code=torch.tensor([[1, 2]]),
        ...     static_numeric_value=torch.tensor([[1., 0.]]),
        ...     static_numeric_value_mask=torch.tensor([[True, False]]),
        ... )

    Similarly to static data, if labels are provided, they must be of shape (batch_size,):

        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     boolean_value=torch.tensor([[True, False], [True, False]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (1,) for boolean_value, but got torch.Size([2, 2])!
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[[1, 2], [3, 0]]]),
        ...     numeric_value=torch.tensor([[[1., 0.], [0., 0.]]]),
        ...     numeric_value_mask=torch.tensor([[[True, False], [False, False]]]),
        ...     time_delta_days=torch.tensor([[1., 2.]]),
        ...     event_mask=torch.tensor([[True, True]]),
        ...     boolean_value=torch.tensor([True]),
        ... )
    """

    PAD_INDEX: ClassVar[int] = 0
    _REQ_TENSORS: ClassVar[list[str]] = ["code", "numeric_value", "numeric_value_mask", "time_delta_days"]

    code: torch.LongTensor | None = None
    numeric_value: torch.FloatTensor | None = None
    numeric_value_mask: torch.BoolTensor | None = None

    time_delta_days: torch.FloatTensor | None = None
    event_mask: torch.BoolTensor | None = None

    static_code: torch.LongTensor | None = None
    static_numeric_value: torch.FloatTensor | None = None
    static_numeric_value_mask: torch.BoolTensor | None = None
    boolean_value: torch.BoolTensor | None = None

    def __check_shape(self, name: str, shape: tuple[int, ...]) -> None:
        """Check that the shape of a tensor matches the expected shape, or raise an appropriate error."""
        got_shape = getattr(self, name).shape
        if got_shape != shape:
            raise ValueError(f"Expected shape {shape} for {name}, but got {got_shape}!")

    def __post_init__(self):
        """Check that the batch is well-formed, raising an error if it is not."""
        for field in fields(self):
            tensor_type = get_args(field.type)[0]
            match value := getattr(self, field.name):
                case None:
                    if field.name in self._REQ_TENSORS:
                        raise ValueError(f"Required tensor {field.name} is missing!")
                    else:
                        pass
                case tensor_type():
                    pass
                case _:
                    raise TypeError(
                        f"Field '{field.name}' expected type {tensor_type}, got type {type(value)}."
                    )

        match self.mode:
            case BatchMode.SEM:
                if self.event_mask is None:
                    raise ValueError(f"Event mask must be provided in {self.mode} mode!")
                self.__check_shape("time_delta_days", self._SE_shape)
                self.__check_shape("event_mask", self._SE_shape)
                self.__check_shape("numeric_value", self._SEM_shape)
                self.__check_shape("numeric_value_mask", self._SEM_shape)
            case BatchMode.SM:
                if self.event_mask is not None:
                    raise ValueError(f"Event mask should not be provided in {self.mode} mode!")
                self.__check_shape("time_delta_days", self._SM_shape)
                self.__check_shape("numeric_value", self._SM_shape)
                self.__check_shape("numeric_value_mask", self._SM_shape)
            case _:  # pragma: no cover
                raise ValueError(f"Invalid mode {self.mode}!")

        if self.has_static:
            if self.static_numeric_value is None or self.static_numeric_value_mask is None:
                raise ValueError("Static numeric value and mask must both be provided if static codes are!")
            if len(self.static_code.shape) != 2 or self.static_code.shape[0] != self.batch_size:
                raise ValueError(
                    f"Expected 2D static data tensors with a matching batch size ({self.batch_size}), "
                    f"but got static_code shape {self.static_code.shape}!"
                )
            self.__check_shape("static_numeric_value", self._static_shape)
            self.__check_shape("static_numeric_value_mask", self._static_shape)
        else:
            if self.static_numeric_value is not None or self.static_numeric_value_mask is not None:
                raise ValueError("Static numeric value and mask should not be provided without codes!")

        if self.has_labels:
            self.__check_shape("boolean_value", (self.batch_size,))

    # Here we define some operators to make this behave like a dictionary:
    def __getitem__(self, key: str) -> torch.Tensor:
        """Get a tensor from the batch by key."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Set a tensor in the batch by key. Only valid if the key is a valid field."""
        raise ValueError("MEDSTorchBatch is immutable!")

    def keys(self) -> Generator[str, None, None]:
        """Get the keys of the batch."""
        for field in fields(self):
            if getattr(self, field.name) is not None:
                yield field.name

    def values(self) -> Generator[torch.Tensor, None, None]:
        """Get the values of the batch."""
        for key in self.keys():
            yield self[key]

    def items(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get the items of the batch."""
        yield from zip(self.keys(), self.values())

    @property
    def mode(self) -> BatchMode:
        """The mode of the batch, reflecting the internal organization of subject measurements."""
        match len(self.code.shape):
            case 2:
                return BatchMode.SM
            case 3:
                return BatchMode.SEM
            case _:
                raise ValueError(
                    "Code shape must have length either 2 (SM mode) or 3 (SEM mode); "
                    f"got shape {self.code.shape}!"
                )

    @property
    def has_static(self) -> bool:
        """Whether the batch has static data."""
        return self.static_code is not None

    @property
    def has_labels(self) -> bool:
        """Whether the batch has labels."""
        return self.boolean_value is not None

    @property
    def batch_size(self) -> int:
        """The number of subjects in the batch."""
        return self.code.shape[0]

    @property
    def max_events_per_subject(self) -> int | None:
        """The maximum number of events for any subject in the batch. Only valid in SEM mode."""
        return self.code.shape[1] if self.mode is BatchMode.SEM else None

    @property
    def max_measurements_per_event(self) -> int | None:
        """The maximum number of measurements for any event in the batch. Only valid in SEM mode."""
        return self.code.shape[2] if self.mode is BatchMode.SEM else None

    @property
    def max_measurements_per_subject(self) -> int | None:
        """The maximum number of measurements for any subject in the batch. Only valid in SM mode."""
        return self.code.shape[1] if self.mode is BatchMode.SM else None

    @property
    def max_static_measurements_per_subject(self) -> int | None:
        """The maximum number of static measurements for any subject in the batch."""
        return self.static_code.shape[1] if self.has_static else None

    @property
    def _SE_shape(self) -> tuple[int, int]:
        """Returns the subject-event shape of the batch. Only valid in SEM mode.

        Examples:
            >>> batch = MEDSTorchBatch(
            ...     time_delta_days=torch.tensor([[1.0, 2.1], [4.0, 0.0]]),
            ...     event_mask=torch.tensor([[True, True], [True, False]]),
            ...     code=torch.tensor([[[1, 2, 3], [3, 0, 0]], [[5, 6, 0], [0, 0, 0]]]),
            ...     numeric_value=torch.tensor(
            ...         [[[1.0, 0.0, -3.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
            ...     ),
            ...     numeric_value_mask=torch.tensor([
            ...         [[True, False, True], [False, False, False]],
            ...         [[False, True, False], [True, True, True]]
            ...     ]), # Note the padding values may be  True or False
            ... )
            >>> print(batch._SE_shape)
            (2, 2)
        """
        return (self.batch_size, self.max_events_per_subject)

    @property
    def _SEM_shape(self) -> tuple[int, int, int]:
        """Returns the subject-event-measurement shape of the batch. Only valid in SEM mode.

        Examples:
            >>> batch = MEDSTorchBatch(
            ...     time_delta_days=torch.tensor([[1.0, 2.1], [4.0, 0.0]]),
            ...     event_mask=torch.tensor([[True, True], [True, False]]),
            ...     code=torch.tensor([[[1, 2, 3], [3, 0, 0]], [[5, 6, 0], [0, 0, 0]]]),
            ...     numeric_value=torch.tensor(
            ...         [[[1.0, 0.0, -3.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
            ...     ),
            ...     numeric_value_mask=torch.tensor([
            ...         [[True, False, True], [False, False, False]],
            ...         [[False, True, False], [True, True, True]]
            ...     ]), # Note the padding values may be  True or False
            ... )
            >>> print(batch._SEM_shape)
            (2, 2, 3)
        """
        return (self.batch_size, self.max_events_per_subject, self.max_measurements_per_event)

    @property
    def _SM_shape(self) -> tuple[int, int]:
        """Returns the subject-measurement shape of the batch. Only valid in SM mode.

        Examples:
            >>> batch = MEDSTorchBatch(
            ...     time_delta_days=torch.tensor([[1.0, 0.0, 0.0, 2.1], [4.0, 0.0, 0.0, 0.0]]),
            ...     code=torch.tensor([[1, 2, 3, 3], [5, 6, 0, 0]]),
            ...     numeric_value=torch.tensor([[1.0, 0.0, -3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False, True, False], [False, True, False, True]]),
            ... )
            >>> print(batch._SM_shape)
            (2, 4)
        """
        return (self.batch_size, self.max_measurements_per_subject)

    @property
    def _static_shape(self) -> tuple[int, int]:
        """Returns the static data shape of the batch. Only valid if the batch has static data.

        Examples:
            >>> batch = MEDSTorchBatch(
            ...     time_delta_days=torch.tensor([[1.0, 0.0, 0.0, 2.1], [4.0, 0.0, 0.0, 0.0]]),
            ...     code=torch.tensor([[1, 2, 3, 3], [5, 6, 0, 0]]),
            ...     numeric_value=torch.tensor([[1.0, 0.0, -3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False, True, False], [False, True, False, True]]),
            ...     static_code=torch.tensor([[1], [5]]),
            ...     static_numeric_value=torch.tensor([[1.0], [0.0]]),
            ...     static_numeric_value_mask=torch.tensor([[True], [True]]),
            ... )
            >>> print(batch._static_shape)
            (2, 1)
        """
        return (self.batch_size, self.max_static_measurements_per_subject)
