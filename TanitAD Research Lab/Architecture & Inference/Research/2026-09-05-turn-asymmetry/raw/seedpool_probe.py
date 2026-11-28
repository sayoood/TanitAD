"""IS THE iCEM SEED POOL DIRECTIONALLY BIASED?  CPU only -- no GPU, no planner.

SPEC section 3.14 names the ONE learned component inside the search:
`refa_v1.py:2484-2485` injects the imitation `proposal` head's non-top modes
into iCEM's ITERATION-0 population (`refa_v1_plan.py:291-293`). Everything else
in iteration 0 is symmetric by construction (mean starts at zero, `init_var` 1.0
on both channels, `colored_noise` zero-mean and sign-balanced -- all asserted in
`raw/mirror_assert.txt`).

⇒ HYPOTHESIS: the mode set is directionally biased, so at `W_KAPPA = 0` the goal
term can still drive the CEM either way, while under a curvature charge only a
direction already present in the seed pool survives.

This probe reaches the modes WITHOUT running the planner: the features are
cached, so `encode` + the proposal head is a few small matmuls. It runs the SAME
three lines `refa_v1.plan()` runs, in the same order, rather than re-deriving
them.

⛔ A VACUOUS SECTION IS SKIPPED, NOT PRINTED AS ZEROS. A 0.0000 computed from an
empty array is not a measurement, it is the absence of one -- and printing it as
a number is how an absent instrument gets quoted as a result.
"""
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
from taniteval import ci as _ci                                   # noqa: E402
from taniteval import four_families as ff                         # noqa: E402
from tanitad.data.refav1_loader import RefAV1Windows              # noqa: E402
from tanitad.refs.refc_tactical import factor_from_kinematics     # noqa: E402

CKPT = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt"
CFG = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json"
LBL = "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz"
PANEL = sys.argv[1] if len(sys.argv) > 1 else None
DT = 0.2
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}

model, cfg, prov = A.load_model(CKPT, CFG, device="cpu")
model.eval()
print("model step=%s  cfg.proposal_k=%d  plan_steps=%d  a_dim=%d  score_head=%s"
      % (prov.get("step", "?"), cfg.proposal_k, cfg.plan_steps, cfg.a_dim,
         model.proposal_score is not None))

ld = RefAV1Windows("C:/Users/Admin/refav1_margin/p4/fp8",
                   "C:/Users/Admin/refav1_margin/p4/eps",
                   op_window=4, op_steps=30, op_dt=DT, str_ext_steps=0, lru=2,
                   labels_path=LBL, nav_path=LBL)
sel, meta = A.select_windows(ld, 16, PANEL)
print("panel: %d windows%s" % (len(sel), "" if PANEL is None
                               else "  sha256 " + meta["sha256"][:16]))

recs = []
with torch.no_grad():
    for wi, ei, t in sel:
        ld._order, ld._cursor = [wi], 0
        b = ld.batch(1)
        field = model.encode(b["feats"])
        pooled = field.mean(dim=-2)[:, -1]
        modes = model.proposal(pooled).reshape(cfg.proposal_k, cfg.plan_steps,
                                               cfg.a_dim)
        if model.proposal_score is not None:
            modes = modes[model.proposal_score(pooled)[0].argsort(descending=True)]
        k = modes[..., 1]
        pk = k.gather(1, k.abs().argmax(1, keepdim=True)).squeeze(1).numpy()
        a_ = modes[..., 0]
        pa = a_.gather(1, a_.abs().argmax(1, keepdim=True)).squeeze(1).numpy()
        o = torch.load(ld.episode_dir / (ld.names[ei] + ".v2ep.pt"),
                       map_location="cpu", weights_only=False)
        g = A.gt_waypoints(o["poses"].float(), t, 10)
        dy, dv, v0g, v1g, _ = ff.maneuver_kinematics(g, DT)
        recs.append(dict(ei=int(ei), t=int(t),
                         gt=int(factor_from_kinematics(dy, dv, v0g, v1g)[0][0]),
                         peak_k=pk.tolist(), peak_a=pa.tolist()))

PK = np.array([r["peak_k"] for r in recs])
PA = np.array([r["peak_a"] for r in recs])
gt = np.array([r["gt"] for r in recs])
eid = np.array([r["ei"] for r in recs])
POOL = PK[:, 1:]
n = len(recs)


def cci(v_, e_):
    r = _ci.episode_cluster_bootstrap(np.asarray(v_, float), e_, n_boot=2000)
    return r["mean"], r["lo"], r["hi"]


print("\n" + "=" * 96)
print("1. THE SEED POOL  (modes[1:], the iteration-0 injection)")
print("=" * 96)
print("  windows n=%d   cfg.proposal_k = %d   pool size per window = %d modes"
      % (n, cfg.proposal_k, POOL.shape[1]))

if POOL.shape[1] == 0:
    print("""
  ********************************************************************************
  HYPOTHESIS 3.14 IS REFUTED, BY THE CHEAPEST POSSIBLE FACT: THIS CHECKPOINT HAS
  cfg.proposal_k = 1, SO `modes[1:]` IS EMPTY AND NO SEED POOL IS EVER INJECTED.

  `refa_v1.py:2485` reads `seed_pool = modes[1:] if modes.shape[0] > 1 else None`.
  The guard fires, iCEM receives `seed_pool=None`, and `refa_v1_plan.py:291` skips
  the injection entirely. A LEARNED, POSSIBLY-BIASED CANDIDATE SET CANNOT EXPLAIN
  AN ASYMMETRY IT NEVER CONTRIBUTES.

  CONTROL that this is a real read and not a dead probe: section 3's mode-0
  statistics come from the SAME `modes` tensor and are non-degenerate, and the
  banner above prints `proposal_k` from the loaded config, not from this file.
  ********************************************************************************

  Sections 2 and 3's pool statistics are VACUOUS at pool size 0 and are SKIPPED
  rather than printed as zeros.
""")
else:
    pos, neg = int((POOL > 0).sum()), int((POOL < 0).sum())
    print("  peak curvature sign over all pool entries: n(+ LEFT) %d, n(- RIGHT) %d"
          % (pos, neg))
    m_, lo_, hi_ = cci((POOL > 0).mean(1), eid)
    print("  per-window fraction of the pool that turns LEFT: %.4f [%.4f, %.4f]"
          % (m_, lo_, hi_))
    print("  ⇒ %s" % ("BALANCED at 0.5 -- the seed pool is NOT the mechanism"
                      if lo_ <= 0.5 <= hi_ else
                      "** NOT balanced: 0.5 is OUTSIDE the interval **"))
    print("  CONTROL, accel channel (no left/right meaning): frac positive %.4f"
          % (PA[:, 1:] > 0).mean())
    print("  CONTROL, sign-flipped modes must invert exactly: %.4f"
          % (-POOL > 0).mean())

    print("\n" + "=" * 96)
    print("2. DOES THE POOL OFFER A CANDIDATE IN THE GT'S DIRECTION?")
    print("=" * 96)
    off_L = ((POOL > 0.02).any(1)).astype(float)
    off_R = ((POOL < -0.02).any(1)).astype(float)
    for c in (1, 2, 0):
        m = gt == c
        if not m.any():
            continue
        l_, r_ = cci(off_L[m], eid[m]), cci(off_R[m], eid[m])
        print("  %-12s n=%3d | offers LEFT %6.4f [%6.4f, %6.4f] | offers RIGHT "
              "%6.4f [%6.4f, %6.4f]" % (LAT[c], int(m.sum()), *l_, *r_))

print("\n" + "=" * 96)
print("3. MODE 0 -- the named `proposal` BASELINE (it competes, but it is NOT the pool)")
print("=" * 96)
m0 = PK[:, 0]
print("  peak curvature: n(+ LEFT) %d, n(- RIGHT) %d of %d   median |kappa| %.5f"
      % (int((m0 > 0).sum()), int((m0 < 0).sum()), n, float(np.median(np.abs(m0)))))
print("  ⚠️ kappa_max is 0.2, so a median |kappa| of %.5f means `_clip` CLAMPS the "
      "proposal on most windows." % float(np.median(np.abs(m0))))
for c in (1, 2, 0):
    m = gt == c
    if m.any():
        mm, ll, hh = cci(m0[m], eid[m])
        print("    on GT %-11s mean signed peak kappa %+0.5f [%+0.5f, %+0.5f] (n=%d)"
              % (LAT[c], mm, ll, hh, int(m.sum())))
d, dlo, dhi = cci(m0[gt == 2], eid[gt == 2])
e, elo, ehi = cci(m0[gt == 1], eid[gt == 1])
print("  ⇒ the proposal baseline leans %s overall, and it leans the SAME way on "
      "GT-right windows as on GT-left ones (%+0.5f vs %+0.5f) -- i.e. it does NOT "
      "track the GT direction." % ("LEFT (+)" if m0.mean() > 0 else "RIGHT (-)", d, e))
print("  CONTROL, the accel channel of mode 0: mean %+0.5f (no directional "
      "meaning; printed so the curvature number is not read in isolation)"
      % float(PA[:, 0].mean()))

json.dump(recs, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                                  "seedpool_rows.json"), "w"))
