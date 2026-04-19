try:
    import lightning

    _HAS_LIGHTNING = True
except ImportError:  # pragma: no cover - exercised in CI envs without the `lightning` extra
    _HAS_LIGHTNING = False

if _HAS_LIGHTNING:
    from .lightning_datamodule import Datamodule
