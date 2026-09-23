"""P1 - does the refcv6 SS6 BOX head's loss path apply ANY visibility filter?

MEASURED, not read. ``refc_v3_train.py:4372`` assembles ``_t3`` straight from
``batch["agent_valid"]`` and calls ``refcv6_perception_branch.box3d_loss_row``.
Neither that function nor ``box3d_head.box3d_set_loss`` mentions a field or a
range cut, and the v6 seam's own ``refc_agents.agent_losses`` applies
``visible_target_filter`` by default at :833.

This probe builds ONE frame carrying three boxes whose visibility is known BY
CONSTRUCTION -- the analytic target the advisory's class F asks for:

  A  (cx=+20, cy=0)     in field, inside the decode box   -> must be supervised
  B  (cx=-20, cy=0)     BEHIND the ego, 180 deg           -> unobservable
  C  (cx=+120, cy=0)    in field, 2x beyond the 60 m box  -> inexpressible

and reads, on the SAME frame:
  * ``box3d_set_loss``   (the refcv6 path) -> n["matched"]
  * ``refc_agents.agent_losses`` (the v6 path) -> n["target_prefilter"] /
    n["target_visible"]

DISCRIMINATING CONTROL (the "both operands fail identically" hole): the same
call is repeated with only box A present. If the refcv6 path filtered, the two
calls would agree; if it does not, they differ by exactly 2.

Run:
  PYTHONPATH=D:/Projects/TanitAD/stack python p1_box_path_filter_probe.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

from tanitad.models.agent_slots import SLOT_SLICES, AGENT_CLASSES
from tanitad.models.box3d_head import (SLOT3D_SLICES, SLOT3D_WIDTH,
                                       Box3DSlotDecoder, box3d_set_loss,
                                       zh_targets)
from tanitad.models.refcv6_perception_branch import box3d_loss_row
from tanitad.refs import refc_agents as RA

OUT = pathlib.Path(__file__).resolve().parents[1] / "raw" / "p1_box_path_filter.json"


def _targets(boxes):
    """boxes: list of (cx, cy, l, w, yaw). Returns the 2-D target dict."""
    n = len(boxes)
    t = {
        "box": torch.tensor([[[b[0], b[1], b[2], b[3]] for b in boxes]],
                            dtype=torch.float32),
        "yaw": torch.tensor([[b[4] for b in boxes]], dtype=torch.float32),
        "cls": torch.zeros(1, n, dtype=torch.long),
        "occ": torch.full((1, n), -1.0),
        "rates": torch.zeros(1, n, 3),
        "rates_mask": torch.zeros(1, n, dtype=torch.bool),
        "valid": torch.ones(1, n, dtype=torch.bool),
    }
    return t


def _pred(n_q=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(1, n_q, SLOT3D_WIDTH, generator=g)
    dec = Box3DSlotDecoder(d_memory=64, n_memory=8, n_queries=n_q,
                           enforce_band=False)
    return dec.decode(raw)


def main() -> int:
    torch.manual_seed(0)
    ALL = [(20.0, 0.0, 4.5, 2.0, 0.0),      # A in field, in box
           (-20.0, 0.0, 4.5, 2.0, 0.0),     # B behind ego
           (120.0, 0.0, 4.5, 2.0, 0.0)]     # C in field, far beyond 60 m
    ONLY_A = ALL[:1]

    res = {"_evidence_class": "MEASURED (ours; this file + raw/p1_box_path_filter.json)",
           "code": {"refcv6_box_loss": "refc_v3_train.py:4372 -> "
                                       "refcv6_perception_branch.box3d_loss_row:503 -> "
                                       "box3d_head.box3d_set_loss:327",
                    "v6_agent_loss": "refc_v3_train.py:3969 -> "
                                     "refc_agents.agent_losses:812 (filter_visible=True) -> "
                                     "refc_agents.visible_target_filter:394"},
           "arms": {}}

    for name, boxes in (("all_three", ALL), ("only_A_control", ONLY_A)):
        pred = _pred(seed=0)
        tgt = zh_targets(_targets(boxes))
        row = box3d_loss_row(pred, tgt)
        # the v6 path, same frame, same targets
        cfg = RA.AgentSeamConfig()
        v6 = RA.agent_losses({k: v for k, v in pred.items()},
                             _targets(boxes), cfg, cam=None)
        res["arms"][name] = {
            "n_boxes_constructed": len(boxes),
            "refcv6_box3d_n_matched": row.get("box3d_n_matched"),
            "refcv6_box3d_n_target": row.get("box3d_n_target"),
            "v6_n_target_prefilter": int(v6["n"]["target_prefilter"]),
            "v6_n_target_visible": int(v6["n"]["target_visible"]),
            "v6_filter_visible": bool(v6["filter_visible"]),
        }

    a = res["arms"]["all_three"]
    b = res["arms"]["only_A_control"]
    res["verdict"] = {
        "refcv6_path_filters": not (a["refcv6_box3d_n_target"]
                                    > b["refcv6_box3d_n_target"]),
        "v6_path_filters": a["v6_n_target_visible"] < a["v6_n_target_prefilter"],
        "delta_refcv6_targets": (a["refcv6_box3d_n_target"]
                                 - b["refcv6_box3d_n_target"]),
        "delta_v6_visible": a["v6_n_target_visible"] - b["v6_n_target_visible"],
        "reading": "if delta_refcv6_targets == 2 and delta_v6_visible == 0, the "
                   "refcv6 box path supervises the behind-ego and 120 m boxes "
                   "that the v6 path drops BY DESIGN",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
