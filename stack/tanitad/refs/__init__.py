"""Reference architectures (REFERENCE_ARCHITECTURES.md) — SEPARATE packages.

REF-A (frozen-DINO world model) and REF-B (E2E baseline) live here, never
entangled with the main model: they import the shared operative predictor /
SigReg / config machinery UNCHANGED and add only their own glue.
"""

from tanitad.refs.refa import (DinoAdapter, FeatureStandardizer, RefAModel,
                               refa_predictor_config)

__all__ = ["DinoAdapter", "FeatureStandardizer", "RefAModel",
           "refa_predictor_config"]
