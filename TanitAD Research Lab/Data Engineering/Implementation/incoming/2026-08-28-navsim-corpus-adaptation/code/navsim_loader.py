"""Reference loader for the NavSim eval corpus — the ONE place the frame choice
is made, so no consumer re-derives geometry.

    frames, mask, frame_obj = load_scene(token, variant="rig_clean")

variant:
  "wide"      256x640, the training frame. Model-input compatible, but 10.7 %
              of pixels are UNOBSERVED (black) -- NavSim's cameras carry 38.5 deg
              of vertical FOV against the frame's 45.3 deg. Use only with the
              returned mask, and stamp the observed fraction on any number.
  "rig_clean" 176x624 == tanitad.data.calib.PHYSICALAI_RIG_CLEAN_176x624,
              MEASURED 100.0000 % observed over all 204 scenes / 5 rigs. The
              exact slice [40:216, 8:632] documented in calib.py. DEFAULT.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data.calib import (  # noqa: E402
    PHYSICALAI_RIG_CLEAN_176x624, PHYSICALAI_WIDE120_256x640)

CORPUS = pathlib.Path(__file__).resolve().parent / "corpus"
#: the centred placement of the rig-clean frame inside the wide one
SLICE = (slice(40, 216), slice(8, 632))


def load_scene(token: str, variant: str = "rig_clean", root: pathlib.Path = CORPUS):
    """-> (frames uint8 [T,H,W,3], observed mask bool [H,W], CanonicalFrame)."""
    if variant not in ("wide", "rig_clean"):
        raise ValueError(f"unknown variant {variant!r}")
    a = np.load(root / "frames" / f"{token}.npy")
    m = np.load(root / "frames" / f"{token}.src.npy") >= 0
    if variant == "wide":
        return a, m, PHYSICALAI_WIDE120_256x640
    a, m = a[:, SLICE[0], SLICE[1]], m[SLICE[0], SLICE[1]]
    if not m.all():                      # the guarantee is checked, not trusted
        raise AssertionError(f"{token}: rig_clean crop has {(~m).sum()} unobserved px")
    return a, m, PHYSICALAI_RIG_CLEAN_176x624


def load_ego(root: pathlib.Path = CORPUS):
    """The EVALUATOR-side sidecar. NEVER feed these columns to a model:
    ego kinematics at inference violate the vision-only rule, and
    `driving_command` is an ORACLE goal input only (declare it if used)."""
    import pandas as pd
    return pd.read_parquet(root / "navsim_ego_sidecar.parquet")
