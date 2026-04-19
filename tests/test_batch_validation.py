"""Validation-path unit tests for `MEDSTorchBatch.__post_init__`.

The batch dataclass treats `numeric_value` and `numeric_value_mask` as an optional
pair — both must be present or both `None`. This module locks in the mismatched-pair
error message since the rest of the test suite only exercises the happy paths (via
`collate`) and the all-off path (via the `include_numeric_value` flag).
"""

from __future__ import annotations

import pytest
import torch

from meds_torchdata.types import MEDSTorchBatch


def test_numeric_value_without_mask_raises():
    """`numeric_value` present while `numeric_value_mask` is `None` is rejected."""
    with pytest.raises(ValueError, match=r"numeric_value and numeric_value_mask must both"):
        MEDSTorchBatch(
            code=torch.zeros((1, 1), dtype=torch.long),
            event_mask=torch.ones((1, 1), dtype=torch.bool),
            numeric_value=torch.zeros((1, 1, 1), dtype=torch.float32),
            # numeric_value_mask intentionally omitted
        )


def test_numeric_value_mask_without_value_raises():
    """`numeric_value_mask` present while `numeric_value` is `None` is rejected."""
    with pytest.raises(ValueError, match=r"numeric_value and numeric_value_mask must both"):
        MEDSTorchBatch(
            code=torch.zeros((1, 1), dtype=torch.long),
            event_mask=torch.ones((1, 1), dtype=torch.bool),
            numeric_value_mask=torch.ones((1, 1, 1), dtype=torch.bool),
            # numeric_value intentionally omitted
        )
