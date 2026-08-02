# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Model abstraction layer for trajectory prediction models.

MODIFIED (TanitAD): the stock Alpamayo/VAM adapters are imported BEHIND A GUARD.

Upstream imports them eagerly, so a missing OPTIONAL model dependency (e.g. the
`alpamayo1_5` package, which we do not install) raises ModuleNotFoundError while merely
importing this package -- taking down every OTHER driver with it, including ours. The
contract classes in `.base` and our own model have no such dependency and must always load.

Behaviour is unchanged when the dependencies ARE installed. A model that fails to import is
recorded in `UNAVAILABLE_MODELS` with its reason, so a missing driver is VISIBLE rather than
silently absent -- asking for it by name still fails loudly at lookup time.
"""

from .base import (
    BaseTrajectoryModel,
    CameraFrame,
    CameraImages,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from .tanitad_model import TanitADModel

__all__ = [
    "BaseTrajectoryModel",
    "CameraFrame",
    "CameraImages",
    "DriveCommand",
    "ModelPrediction",
    "PredictionInput",
    "TanitADModel",
    "UNAVAILABLE_MODELS",
]

UNAVAILABLE_MODELS: dict[str, str] = {}

for _name, _mod in (("Alpamayo15Model", ".alpamayo1_5_model"),
                    ("Alpamayo1Model", ".alpamayo1_model"),
                    ("ManualModel", ".manual_model"),
                    ("VAMModel", ".vam_model")):
    try:
        _m = __import__(f"alpasim_driver.models{_mod}", fromlist=[_name])
        globals()[_name] = getattr(_m, _name)
        __all__.append(_name)
    except Exception as _e:                      # optional dependency absent
        UNAVAILABLE_MODELS[_name] = f"{type(_e).__name__}: {_e}"
