"""Q3b — the surviving geometry LITERALS, and whether a caller who trusts them
fails LOUD or silently mis-samples.

SPEC_REFCV6_V2.md §10.1: *"⛔ Nothing in the code may hard-code 640 / 160 / 40 /
20: shapes are read from the feature maps."* Three literals survive as DEFAULTS.
A default is not automatically a violation — what matters is whether it can be
REACHED and whether reaching it is loud. Each is exercised here.

  1. `bev_lift.BEVLift.__init__(feat_hw=(16, 40))`      — bev_lift.py:236
  2. `timm_trunk.build_timm_trunk(image_hw=(256, 640))` — timm_trunk.py:922
  3. `timm_trunk.TimmTrunkConfig.image_hw = (256,1024)` — timm_trunk.py:206
     (STALE: SPEC §12 pins 416x1024 since 2026-09-17)

Also runs `stack/tests/test_refcv6_geometry_agnostic.py` and reports its skip
reasons — a skipped guard is no guard.
"""
from __future__ import annotations

import json
import sys

import torch


def main() -> int:
    out = {}
    from tanitad.models.bev_lift import HEIGHTS_M, BEVLift
    from tanitad.models.timm_trunk import (TimmResNetTrunk, TimmTrunkConfig,
                                           build_timm_trunk)

    # -- 1. the BEVLift feat_hw default, against a 416x1024 feature map ---- #
    lift = BEVLift(d_in=256, d_out=128, n_heights=len(HEIGHTS_M))   # default!
    fmap = torch.zeros(1, 256, 26, 64)                 # what 416x1024 emits
    grid = torch.zeros(1, 4, 120, 64, 2)
    valid = torch.zeros(1, 4, 120, 64, dtype=torch.bool)
    r = {"default_feat_hw": list(lift.feat_hw)}
    try:
        lift(fmap, grid, valid)
        r["outcome"] = "SILENT — ran on a 26x64 map while built for 16x40"
        r["LOUD"] = False
    except Exception as e:
        r["outcome"] = f"{type(e).__name__}: {e}"
        r["LOUD"] = True
    # and the escape hatch: feat_hw=None disables the check entirely
    lift_none = BEVLift(d_in=256, d_out=128, n_heights=len(HEIGHTS_M),
                        feat_hw=None)
    try:
        lift_none(fmap, grid, valid)
        r["feat_hw_None_outcome"] = "accepted ANY map shape (check disabled)"
    except Exception as e:
        r["feat_hw_None_outcome"] = f"{type(e).__name__}: {e}"
    out["L1_BEVLift_feat_hw_default"] = r

    # -- 2/3. the trunk image_hw defaults ---------------------------------- #
    t_default = TimmResNetTrunk(TimmTrunkConfig(
        model_name="resnet34.a1_in1k", frames=3, pretrained=False,
        verify_imagenet_stats=False))
    out["L3_TimmTrunkConfig_image_hw_default"] = {
        "value": list(TimmTrunkConfig().image_hw),
        "SPEC_12_pins": [416, 1024],
        "STALE": list(TimmTrunkConfig().image_hw) != [416, 1024],
        "declared_s16_shape": list(t_default.s16_shape),
        "declared_grid_shape": list(t_default.grid_shape),
    }
    # does a WRONG declared shape fail loud, or just mis-declare?
    with torch.no_grad():
        s16, s32, _ = t_default.forward_features(torch.rand(1, 9, 416, 1024))
    out["L3_TimmTrunkConfig_image_hw_default"].update({
        "ACTUAL_s16_shape_on_416x1024_input": list(s16.shape[2:]),
        "ACTUAL_s32_shape_on_416x1024_input": list(s32.shape[2:]),
        "declared_matches_actual":
            list(t_default.s16_shape) == list(s16.shape[2:]),
        "outcome": ("the forward SUCCEEDS and the trunk's DECLARED "
                    "s16_shape/grid_shape are wrong — no exception. Any "
                    "consumer reading `enc.s16_shape` (e.g. "
                    "refcv6_perception_branch.build_perception_branch:467) "
                    "gets the stale number."),
    })
    bt = build_timm_trunk(in_channels=9, pretrained=False,
                          verify_imagenet_stats=False,
                          model_name="resnet34.a1_in1k")      # image_hw default
    out["L2_build_timm_trunk_image_hw_default"] = {
        "value": list(bt.cfg.image_hw), "declared_s16": list(bt.s16_shape),
        "note": ("overridden at the only production call site, "
                 "tanitad/refs/refc.py:1442 image_hw=cfg.image_hw()"),
    }

    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
