"""Tokenization stage rejects `train_only=True`.

Not an @Stage.register test case because the error fires before any shard
iteration / output is produced; go straight at the `main` function.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from meds_torchdata.preprocessing.tokenization.tokenization import main


def test_tokenization_rejects_train_only(tmp_path):
    """`train_only=True` on the tokenization stage is unsupported and must raise."""
    cfg = OmegaConf.create(
        {
            "stage": "tokenization",
            "stage_cfg": {
                "output_dir": str(tmp_path),
                "train_only": True,
                "data_input_dir": str(tmp_path),
            },
            "input_dir": str(tmp_path),
            "output_dir": str(tmp_path),
        }
    )
    with pytest.raises(ValueError, match=r"train_only=True is not supported for this stage"):
        # `main` is wrapped by `@Stage.register`; access the underlying function if needed.
        inner = getattr(main, "__wrapped__", main)
        inner(cfg)
