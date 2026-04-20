try:
    import lightning

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False

if _HAS_LIGHTNING:
    from .lightning_datamodule import Datamodule
