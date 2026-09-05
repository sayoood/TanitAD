"""THE WIDE PANEL'S DECODED-GOAL CLUSTER STRUCTURE, before the arms land.

⛔ WHY THIS IS NOT A LEAK. The decoded goal token is an INPUT to the planner, not
its output: it is `lat_head(intent)` from vision + nav + v0, and it is identical
across every banked arm (40/40) INCLUDING across plan seeds. Reading it is panel
characterisation of exactly the kind SPEC section 1's cluster criterion demands.
Nothing here touches `cl`, a cost, or a plan.

⭐ WHY IT MATTERS. On the banked 40-window panel the two turn-goal strata lived
in COMPLETELY DISJOINT episodes ({1,6} vs {0,2,4,7}), which made the retention
statistic structurally unattributable. If the wide panel repeats that, the
retention read is void there too -- and I need to know BEFORE reading it, not
after.

CPU only: the features are cached, so encode + the brains + one head is a few
small matmuls. Reproduces refa_v1.plan()'s own lines, in its own order.
"""
import collections
import json
import os
import sys

import numpy as np
import torch

WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT, WT + "/taniteval/tools"):
    if p not in sys.path:
        sys.path.insert(0, p)
import refav1_arm as A                                            # noqa: E402
from taniteval import four_families as ff                         # noqa: E402
from tanitad.data.refav1_loader import RefAV1Windows              # noqa: E402
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV   # noqa: E402
from tanitad.refs.refc_tactical import factor_from_kinematics     # noqa: E402

CKPT = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt"
CFG = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json"
LBL = "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz"
PANEL = sys.argv[1] if len(sys.argv) > 1 else None
DT = 0.2
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}
I_TL, I_TR = LATV.index("TURN_L"), LATV.index("TURN_R")

model, cfg, prov = A.load_model(CKPT, CFG, device="cpu")
model.eval()
ld = RefAV1Windows("C:/Users/Admin/refav1_margin/p4/fp8",
                   "C:/Users/Admin/refav1_margin/p4/eps",
                   op_window=4, op_steps=30, op_dt=DT, str_ext_steps=0, lru=2,
                   labels_path=LBL, nav_path=LBL)
sel, meta = A.select_windows(ld, 16, PANEL)
print("panel: %d windows%s" % (len(sel), "" if PANEL is None
                               else "  sha256 " + meta["sha256"][:16]))

rows = []
with torch.no_grad():
    for wi, ei, t in sel:
        ld._order, ld._cursor = [wi], 0
        b = ld.batch(1)
        nav = b.get("nav_cmd")
        field = model.encode(b["feats"])
        brains = model._run_brains(field.mean(dim=-2), nav)
        lat = int(model.lat_head(brains["intent"]).argmax(-1)[0])
        o = torch.load(ld.episode_dir / (ld.names[ei] + ".v2ep.pt"),
                       map_location="cpu", weights_only=False)
        g = A.gt_waypoints(o["poses"].float(), t, 10)
        dy, dv, v0g, v1g, _ = ff.maneuver_kinematics(g, DT)
        rows.append(dict(ei=int(ei), t=int(t), goal=lat,
                         gt=int(factor_from_kinematics(dy, dv, v0g, v1g)[0][0])))

goal = np.array([r["goal"] for r in rows])
gt = np.array([r["gt"] for r in rows])
eid = np.array([r["ei"] for r in rows])

print("\n" + "=" * 92)
print("DECODED GOAL TOKEN on the wide panel (an INPUT to the planner)")
print("=" * 92)
c = collections.Counter(LATV[x] for x in goal)
print("  counts: %s" % dict(c))
gL, gR = goal == I_TL, goal == I_TR
epL = sorted(set(eid[gL].tolist()))
epR = sorted(set(eid[gR].tolist()))
both = sorted(set(epL) & set(epR))
print("  TURN_L n=%2d over %d episodes %s" % (gL.sum(), len(epL), epL))
print("  TURN_R n=%2d over %d episodes %s" % (gR.sum(), len(epR), epR))
print("  episodes carrying BOTH turn goals: %s" % (both or "NONE"))
print()
if both:
    nl = int((gL & np.isin(eid, both)).sum())
    nr = int((gR & np.isin(eid, both)).sum())
    print("  ⭐ THE DEGENERACY IS BROKEN: %d episodes carry both turn goals, with"
          % len(both))
    print("     %d TURN_L and %d TURN_R windows inside them, so the RETENTION"
          % (nl, nr))
    print("     statistic has a within-episode contrast on this panel.")
    print("     (banked 40-window panel: ZERO episodes carried both.)")
else:
    print("  ⛔ STILL DEGENERATE: no episode carries both turn goals, so the")
    print("     retention statistic is UNATTRIBUTABLE on this panel too and must")
    print("     be reported as such. The RECALL statistic is unaffected -- its")
    print("     strata are GT-defined and 4 episodes carry both directions.")
print()
print("  POWER against SPEC section 1 (>= 27 windows, >= 5 clusters):")
for nm, m, ep in (("goal TURN_L", gL, epL), ("goal TURN_R", gR, epR)):
    print("    %-12s n=%2d %-10s clusters=%d %s"
          % (nm, int(m.sum()), "MET" if m.sum() >= 27 else "**MISSED**",
             len(ep), "MET" if len(ep) >= 5 else "**MISSED**"))

print("\n  goal-vs-GT confusion (GT rows, decoded-goal cols):")
print("        %10s %10s %10s %8s" % ("LANE_KEEP", "TURN_L", "TURN_R", "other"))
for cix, nm in ((0, "GT LK"), (1, "GT TL"), (2, "GT TR")):
    m = gt == cix
    lk = int((goal[m] == LATV.index("LANE_KEEP")).sum())
    tl = int((goal[m] == I_TL).sum())
    tr = int((goal[m] == I_TR).sum())
    print("  %-5s %10d %10d %10d %8d" % (nm, lk, tl, tr, int(m.sum()) - lk - tl - tr))
print("  ** GOAL RECALL (the head alone, before any cost): left %d/%d = %.4f, "
      "right %d/%d = %.4f **"
      % (int((goal[gt == 1] == I_TL).sum()), int((gt == 1).sum()),
         float((goal[gt == 1] == I_TL).mean()),
         int((goal[gt == 2] == I_TR).sum()), int((gt == 2).sum()),
         float((goal[gt == 2] == I_TR).mean())))
print("  CONTROL, GT strata on the selected windows: %s"
      % dict(collections.Counter(LAT[x] for x in gt)))

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                                  "goaldecode_rows.json"), "w"))
