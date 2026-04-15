"""Histogram and boundary tests for `SubsequenceSamplingStrategy`.

The single-sample doctests in `src/meds_torchdata/types.py` already cover the support,
reachability, and `None`-on-fit semantics of each strategy. This file complements them with
*empirical* per-event inclusion histograms that verify the statistical shape of each
strategy's draws over many samples:

- `BALANCED_RANDOM` must produce a **flat** inclusion histogram (`max_seq_len / (seq_len +
  max_seq_len - 1)` per position), which is the defining property from issue #67.
- `RANDOM` must produce the canonical **trapezoid**: linear ramp up over the first
  `max_seq_len - 1` positions, flat plateau of `max_seq_len / (seq_len - max_seq_len + 1)`
  through the middle, linear ramp back down over the last `max_seq_len - 1` positions.

The number of draws used for every histogram is configurable via the `--n-sampling-draws`
pytest option (default 20k) so tolerance-based asserts can be loosened or tightened without
touching the tests.
"""

from collections import Counter

import numpy as np
import pytest

from meds_torchdata.types import SubsequenceSamplingStrategy


def inclusion_histogram(
    strategy: str | SubsequenceSamplingStrategy,
    seq_len: int,
    max_seq_len: int,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the empirical per-event inclusion rate array from `n_draws` samples.

    The returned array has length `seq_len`. Element `i` is the fraction of draws whose
    resulting window `[max(0, st), min(seq_len, st + max_seq_len))` contains position `i`.
    """

    counts = np.zeros(seq_len, dtype=np.int64)
    for _ in range(n_draws):
        st = SubsequenceSamplingStrategy.subsample_st_offset(strategy, seq_len, max_seq_len, rng=rng)
        if st is None:
            st = 0
        start = max(0, st)
        end = min(seq_len, st + max_seq_len)
        counts[start:end] += 1
    return counts / n_draws


def trapezoidal_theoretical_rates(seq_len: int, max_seq_len: int) -> np.ndarray:
    """Closed-form per-event inclusion rates for `RANDOM` sampling.

    For `st ~ Uniform([0, seq_len - max_seq_len])` the number of windows covering position
    `i` is `min(i + 1, max_seq_len, seq_len - i)`, normalized by the `seq_len - max_seq_len
    + 1` uniform start offsets. The resulting profile is a trapezoid: linear ramp up over
    the first `max_seq_len - 1` positions, flat plateau through the middle, linear ramp
    down over the last `max_seq_len - 1` positions.
    """

    n_starts = seq_len - max_seq_len + 1
    return np.asarray(
        [min(i + 1, max_seq_len, seq_len - i) / n_starts for i in range(seq_len)],
        dtype=np.float64,
    )


@pytest.mark.parametrize(("seq_len", "max_seq_len"), [(20, 5), (50, 10), (100, 25)])
def test_balanced_random_histogram_is_uniform(seq_len: int, max_seq_len: int, n_sampling_draws: int):
    """`BALANCED_RANDOM` must produce a flat per-event inclusion rate."""

    rng = np.random.default_rng(0xBA1A0CED)
    empirical = inclusion_histogram("balanced_random", seq_len, max_seq_len, n_sampling_draws, rng)
    expected = max_seq_len / (seq_len + max_seq_len - 1)

    max_abs_err = float(np.max(np.abs(empirical - expected)))
    assert max_abs_err < 0.02, (
        f"BALANCED_RANDOM histogram is not flat (max deviation {max_abs_err:.4f} from "
        f"expected uniform rate {expected:.4f} with {n_sampling_draws} draws); "
        f"empirical rates: {empirical.tolist()}"
    )

    # Make the boundary-vs-middle contrast explicit — the whole point of BALANCED_RANDOM is
    # that the rates at positions 0 and L-1 are indistinguishable from the plateau.
    boundary = 0.5 * (empirical[0] + empirical[-1])
    middle = float(np.mean(empirical[max_seq_len : seq_len - max_seq_len]))
    assert abs(boundary - middle) < 0.02


@pytest.mark.parametrize(("seq_len", "max_seq_len"), [(20, 5), (50, 10), (100, 25)])
def test_random_histogram_is_trapezoidal(seq_len: int, max_seq_len: int, n_sampling_draws: int):
    """`RANDOM` must match the trapezoidal closed form — boundaries under, middle plateau."""

    rng = np.random.default_rng(0xBA1A0CED)
    empirical = inclusion_histogram("random", seq_len, max_seq_len, n_sampling_draws, rng)
    expected = trapezoidal_theoretical_rates(seq_len, max_seq_len)

    max_abs_err = float(np.max(np.abs(empirical - expected)))
    assert max_abs_err < 0.02, (
        f"RANDOM histogram does not match the trapezoidal closed form "
        f"(max deviation {max_abs_err:.4f} with {n_sampling_draws} draws); "
        f"empirical rates: {empirical.tolist()}, expected: {expected.tolist()}"
    )

    # The defining structural bias: boundary positions are sampled much less than the
    # plateau. This contrast is what BALANCED_RANDOM was created to close.
    boundary = 0.5 * (empirical[0] + empirical[-1])
    middle = float(np.mean(empirical[max_seq_len : seq_len - max_seq_len]))
    assert boundary < 0.5 * middle, (
        f"RANDOM should undersample boundary positions relative to the plateau, but got "
        f"boundary={boundary:.4f}, middle={middle:.4f}"
    )


def test_balanced_random_process_dynamic_data_histogram(n_sampling_draws: int):
    """End-to-end integration: plug `balanced_random` through `process_dynamic_data`.

    Verifies both the set of reachable events and the empirical histogram shape after the
    full `JointNestedRaggedTensorDict` flatten → slice pipeline used at training time. The
    histogram check catches regressions where (e.g.) the negative-start clamp is dropped or
    an off-by-one slips back in.
    """

    from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

    from meds_torchdata.config import MEDSTorchDataConfig

    codes = list(range(20))
    data = JointNestedRaggedTensorDict({"time_delta": codes, "code": [[c] for c in codes]})
    max_seq_len = 5

    cfg = MEDSTorchDataConfig(".", max_seq_len=max_seq_len, seq_sampling_strategy="balanced_random")

    length_counts: Counter[int] = Counter()
    position_counts = np.zeros(len(codes), dtype=np.int64)
    for seed in range(n_sampling_draws):
        out = cfg.process_dynamic_data(data, rng=seed).to_dense()
        out_codes = [int(x) for x in out["code"]]
        length_counts[len(out_codes)] += 1
        # Because the `code` payload at position `i` is literally the integer `i` by
        # construction, the returned codes *are* the original positions — use them to
        # accumulate the histogram directly.
        for c in out_codes:
            position_counts[c] += 1

    # Reachability: every code must have been sampled at least once.
    assert np.all(position_counts > 0), (
        f"Some codes were never sampled through process_dynamic_data: {position_counts.tolist()}"
    )

    # Histogram shape: uniform per-position inclusion up to statistical noise. Expected
    # rate is the same `max_seq_len / (seq_len + max_seq_len - 1)` formula, but here
    # `seq_len` is the post-flatten measurement count (which equals `len(codes)` for this
    # fixture because each event has exactly one measurement).
    rate = position_counts / n_sampling_draws
    expected = max_seq_len / (len(codes) + max_seq_len - 1)
    max_abs_err = float(np.max(np.abs(rate - expected)))
    assert max_abs_err < 0.02, (
        f"process_dynamic_data + balanced_random histogram is not flat "
        f"(max deviation {max_abs_err:.4f} from expected {expected:.4f}); rates: "
        f"{rate.tolist()}"
    )

    # Variable-length windows: windows overhanging the boundary return fewer than
    # `max_seq_len` elements, so the distribution of returned lengths must be non-trivial.
    assert len(length_counts) > 1, (
        f"Expected variable-length windows at boundaries, got {dict(length_counts)}"
    )
