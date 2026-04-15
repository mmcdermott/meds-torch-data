"""Tests for the BALANCED_RANDOM subsequence sampling strategy (issue #67)."""

from collections import Counter

import numpy as np
import pytest

from meds_torchdata.types import SubsequenceSamplingStrategy


@pytest.mark.parametrize(
    ("seq_len", "max_seq_len"),
    [(10, 3), (25, 7), (50, 1), (100, 20)],
)
def test_balanced_random_support_is_full(seq_len: int, max_seq_len: int):
    """Every offset in `{-(W-1), ..., L-1}` must be reachable."""

    seen = {
        SubsequenceSamplingStrategy.subsample_st_offset("balanced_random", seq_len, max_seq_len, rng=seed)
        for seed in range(20_000)
    }
    expected = set(range(-(max_seq_len - 1), seq_len))
    assert seen == expected


def test_balanced_random_returns_none_when_no_subsampling():
    for seq_len, max_seq_len in [(3, 5), (5, 5), (0, 5)]:
        result = SubsequenceSamplingStrategy.subsample_st_offset(
            "balanced_random", seq_len, max_seq_len, rng=0
        )
        assert result is None, f"seq_len={seq_len}, max_seq_len={max_seq_len} should skip sampling"


@pytest.mark.parametrize(
    ("seq_len", "max_seq_len"),
    [(20, 5), (50, 10), (100, 25)],
)
def test_balanced_random_per_event_inclusion_is_uniform(seq_len: int, max_seq_len: int):
    """Empirically verify uniform per-event inclusion probability.

    For each position `i` the inclusion probability should be exactly
    `max_seq_len / (seq_len + max_seq_len - 1)`, independent of `i`. We draw a large
    number of windows and check that every position's empirical rate lies within a
    tight tolerance of the theoretical rate. This is the property that distinguishes
    `BALANCED_RANDOM` from `RANDOM` (which produces a trapezoidal distribution).
    """

    rng = np.random.default_rng(0xBA1A0CED)
    n_draws = 20_000

    inclusion_counts = np.zeros(seq_len, dtype=np.int64)
    for _ in range(n_draws):
        st = SubsequenceSamplingStrategy.subsample_st_offset(
            "balanced_random",
            seq_len,
            max_seq_len,
            rng=rng,
        )
        start = max(0, st)
        end = min(seq_len, st + max_seq_len)
        inclusion_counts[start:end] += 1

    empirical = inclusion_counts / n_draws
    expected = max_seq_len / (seq_len + max_seq_len - 1)

    # Every position has the same expected rate -> tight absolute tolerance.
    max_abs_err = float(np.max(np.abs(empirical - expected)))
    assert max_abs_err < 0.02, (
        f"Max deviation {max_abs_err:.4f} from expected uniform rate {expected:.4f} "
        f"exceeded tolerance; empirical inclusion rates: {empirical.tolist()}"
    )

    # Make the boundary-vs-middle contrast explicit: the whole point of BALANCED_RANDOM
    # is that boundary positions should *not* be underrepresented relative to the plateau.
    boundary_rate = 0.5 * (empirical[0] + empirical[-1])
    middle_rate = float(np.mean(empirical[max_seq_len : seq_len - max_seq_len]))
    assert abs(boundary_rate - middle_rate) < 0.02


def test_balanced_random_differs_from_random_at_boundaries():
    """Sanity check: vanilla RANDOM undersamples boundaries; BALANCED_RANDOM does not."""

    seq_len, max_seq_len = 50, 10
    n_draws = 20_000

    random_counts = np.zeros(seq_len, dtype=np.int64)
    balanced_counts = np.zeros(seq_len, dtype=np.int64)
    rng_random = np.random.default_rng(0)
    rng_balanced = np.random.default_rng(0)

    for _ in range(n_draws):
        st_r = SubsequenceSamplingStrategy.subsample_st_offset("random", seq_len, max_seq_len, rng=rng_random)
        random_counts[st_r : st_r + max_seq_len] += 1

        st_b = SubsequenceSamplingStrategy.subsample_st_offset(
            "balanced_random", seq_len, max_seq_len, rng=rng_balanced
        )
        balanced_counts[max(0, st_b) : min(seq_len, st_b + max_seq_len)] += 1

    random_rate = random_counts / n_draws
    balanced_rate = balanced_counts / n_draws

    # Under RANDOM, position 0 appears in exactly 1/(L - W + 1) of windows -- much smaller
    # than the plateau rate of W/(L - W + 1). Under BALANCED_RANDOM the two are equal.
    assert random_rate[0] < 0.5 * random_rate[seq_len // 2]
    assert balanced_rate[0] > 0.8 * balanced_rate[seq_len // 2]


def test_balanced_random_with_process_dynamic_data():
    """End-to-end check through `MEDSTorchDataConfig.process_dynamic_data`."""

    from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

    from meds_torchdata.config import MEDSTorchDataConfig

    code = list(range(20))
    data = JointNestedRaggedTensorDict({"time_delta": code, "code": [[c] for c in code]})

    cfg = MEDSTorchDataConfig(
        ".",
        max_seq_len=5,
        seq_sampling_strategy="balanced_random",
    )

    lengths = Counter()
    seen_code_at_position = dict.fromkeys(code, 0)
    for seed in range(2000):
        out = cfg.process_dynamic_data(data, rng=seed).to_dense()
        out_codes = [int(x) for x in out["code"]]
        lengths[len(out_codes)] += 1
        for c in out_codes:
            seen_code_at_position[c] += 1

    # Every code must be reachable (this exact assertion failed for `RANDOM` before #67).
    assert all(v > 0 for v in seen_code_at_position.values()), (
        f"Some events were never sampled: {seen_code_at_position}"
    )

    # Windows at the boundaries are shorter than max_seq_len -> we should see a distribution
    # of returned sequence lengths, not a single constant length.
    assert len(lengths) > 1, f"Expected variable-length windows at boundaries, got {dict(lengths)}"
