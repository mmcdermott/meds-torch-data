"""Smoke test for the optional-extension import fallback in `meds_torchdata.extensions`.

The `extensions/__init__.py` has a `try: import lightning / except ImportError:` gate
so the package still imports cleanly when the `lightning` extra isn't installed. In CI
this gets exercised by the job that runs without `--extra lightning`; this test lets
single-pass local coverage also hit the except branch without needing to uninstall
`lightning`.
"""

from __future__ import annotations

import builtins
import importlib
import sys


def test_extensions_without_lightning(monkeypatch):
    """Simulating `ImportError: lightning` leaves `_HAS_LIGHTNING = False`."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightning" or name.startswith("lightning."):
            raise ImportError(f"Simulated absence of {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("meds_torchdata.extensions", None)
    sys.modules.pop("lightning", None)

    extensions = importlib.import_module("meds_torchdata.extensions")
    assert extensions._HAS_LIGHTNING is False
    assert not hasattr(extensions, "Datamodule")
