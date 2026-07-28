"""Per-regime trace slicing + interpretation (decode vs prefill), isolated so a change to
one cannot affect the other. See regimes/base.py for the contract."""
from .base import Regime, Window
from .decode import DecodeRegime
from .prefill import PrefillRegime


def regime_for(mode):
    """Return the Regime instance for a mode string ('decode' | 'prefill')."""
    return PrefillRegime() if mode == "prefill" else DecodeRegime()


__all__ = ["Regime", "Window", "DecodeRegime", "PrefillRegime", "regime_for"]
