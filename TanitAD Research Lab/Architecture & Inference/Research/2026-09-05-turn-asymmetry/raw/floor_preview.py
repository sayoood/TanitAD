"""THE WIDE PANEL'S FLOOR ASYMMETRY, measured BEFORE any planner runs.

`ha`, `ha0`, `ha0_ext` and `ol` are pure kinematics on the RECORDED actions -- no
model, no GPU, no planner. So the panel's corpus-plus-labeller directional
baseline can be established up front, and it is the number every planner result
must be read against.

⛔ It also cannot leak the outcome: nothing here depends on `cl`.

Uses the ARM TOOL'S OWN functions (`hold_action_controls`, `hold_ext_controls`,
`paths_from_controls`, `gt_waypoints`) rather than a re-implementation -- a
control re-implemented beside the harness that uses it is a control that can
drift away from it.
"""
import collections
import json
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

PANEL = sys.argv[1]
K, DT = 10, 0.2
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}

ld = RefAV1Windows("C:/Users/Admin/refav1_margin/p4/fp8",
                   "C:/Users/Admin/refav1_margin/p4/eps",
                   op_window=4, op_steps=30, op_dt=DT, str_ext_steps=0, lru=8)
sel, meta = A.select_windows(ld, 16, PANEL)
base, _ = A.select_windows(ld, 16, None)
print("panel %s: %d windows (sha256 %s)  |  CONTROL stride-16 path: %d windows"
      % (PANEL.split("/")[-1], len(sel), meta["sha256"][:16], len(base)))

rows = {"g": [], "ha": [], "ha0": [], "ha0_ext": [], "ol": [], "eid": [], "v0": []}
for wi, ei, t in sel:
    nm = ld.names[ei]
    feats, v, kap = ld._episode(nm)
    poses = torch.load(ld.episode_dir / (nm + ".v2ep.pt"), map_location="cpu",
                       weights_only=False)["poses"].float()
    v0 = float(v[2 * t])
    g = A.gt_waypoints(poses, t, K)[0]
    ha = A.hold_action_controls(ld, v, kap, t)[None].expand(K, 2).float()
    hx = A.hold_ext_controls(ld, v, kap, t)[None].expand(K, 2).float()
    h0 = A.hold_v0_controls(K)
    ol = ld._kin_actions(v, kap, t, K).float()
    for k, c in (("ha", ha), ("ha0", h0), ("ha0_ext", hx), ("ol", ol)):
        rows[k].append(A.paths_from_controls(c[None], v0, DT, K)[0].numpy())
    rows["g"].append(g.numpy())
    rows["eid"].append(ei)
    rows["v0"].append(v0)

G = np.stack(rows["g"])
eid = np.array(rows["eid"])


def lab(X):
    t_ = torch.as_tensor(np.stack(X) if isinstance(X, list) else X).float()
    dy, dv, a0, a1, _ = ff.maneuver_kinematics(t_, DT)
    return factor_from_kinematics(dy, dv, a0, a1)[0].numpy()


lat_g = lab(G)
mL, mR = lat_g == 1, lat_g == 2
c = collections.Counter(LAT[x] for x in lat_g)
print("\nGT on the selected windows: %s" % dict(c))
print("  ⭐ CONTROL, must match the panel file's own strata: %s"
      % json.load(open(PANEL))["strata"])
print("  episodes: turn_left %s | turn_right %s | carrying BOTH %s"
      % (sorted(set(eid[mL].tolist())), sorted(set(eid[mR].tolist())),
         sorted(set(eid[mL].tolist()) & set(eid[mR].tolist()))))


def cci(v_, e_):
    r = _ci.episode_cluster_bootstrap(np.asarray(v_, float), e_, n_boot=2000)
    return r["mean"], r["lo"], r["hi"]


def within(hit, mask_a, mask_b, seed=0):
    eps = sorted(set(eid[mask_a].tolist()) & set(eid[mask_b].tolist()))
    if not eps:
        return float("nan"), float("nan"), float("nan"), []
    per = {e: float(hit[mask_a & (eid == e)].mean() - hit[mask_b & (eid == e)].mean())
           for e in eps}
    rng = np.random.default_rng(seed)
    d = [float(np.mean([per[e] for e in rng.choice(eps, len(eps))])) for _ in range(2000)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(list(per.values()))), float(lo), float(hi), eps


print("\n" + "=" * 96)
print("THE FLOORS' OWN DIRECTIONAL BASELINE ON THE WIDE PANEL (no planner, no GPU)")
print("=" * 96)
print("  %-9s | %-24s | %-24s | %-24s | within-episode"
      % ("arm", "recall LEFT", "recall RIGHT", "pooled R - L"))
for a_ in ("ha", "ha0", "ha0_ext", "ol"):
    hit = (lab(rows[a_]) == lat_g).astype(float)
    lm, ll, lh = cci(hit[mL], eid[mL])
    rm, rl, rh = cci(hit[mR], eid[mR])
    p, plo, phi, _ = within(hit, mR, mL)
    d = rm - lm
    print("  %-9s | %6.4f [%6.4f, %6.4f] | %6.4f [%6.4f, %6.4f] | %+6.4f          "
          "  | %+6.4f [%+6.4f, %+6.4f]" % (a_, lm, ll, lh, rm, rl, rh, d, p, plo, phi))
print("  CONTROL, GT against itself (must be exactly 1.0000 / 1.0000): "
      "%.4f / %.4f" % ((lab(G) == lat_g)[mL].mean(), (lab(G) == lat_g)[mR].mean()))
print("\n  ⇒ this is the directional baseline the planner arms must be read "
      "against. A planner gap no larger than these is the CORPUS, not the cost.")
