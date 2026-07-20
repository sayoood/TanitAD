"""SC-13 probe, follow-up: is the anticipation AUROC just a SPEED confound?

The first pass found braking anchors sit at median v0 8.94 m/s while the cruise
controls sit at 17.34 m/s. Any signal that merely grows with low speed would
score high without anticipating anything. This re-scores the SAME saved windows
with the speed difference removed, two independent ways:

  1. STRATIFIED — AUROC inside v0 bins, pooled weighted by event count.
  2. MATCHED    — for each event anchor, keep only cruise anchors within
                  +/- TOL m/s of its v0 (a per-event matched control set), then
                  pool the pairwise comparisons.

Same arms as the parent probe: informed (leaks) / held / blind / reactive.
"""
from __future__ import annotations

import json
import sys

import torch

P = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/taniteval/results/sc13_flagship30k_windows.pt"
TOL = 1.0          # m/s speed-matching tolerance
DROP, DROP_FAR, VMIN = 2.0, 1.5, 5.0

p = torch.load(P, map_location="cpu", weights_only=False)
v0, vf = p["v0"], p["vfut"]
drop_near = v0 - vf[:, :20].min(dim=1).values
drop_far = v0 - vf[:, 20:30].min(dim=1).values
swing = (vf - v0[:, None]).abs().max(dim=1).values
fast = v0 >= VMIN
LAB = {"brake_near": fast & (drop_near >= DROP),
       "brake_far": fast & (drop_far >= DROP_FAR) & (drop_near < 0.75)}
cruise = fast & (swing <= 0.5)

sig = {a: p["cv"][:, 0] - p[a][:, 0] for a in ("informed", "held", "blind")}
sig["gt_oracle"] = p["cv"][:, 0] - p["gt"][:, 0]
sig["reactive"] = p["reactive"]


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = pos.reshape(-1, 1) > neg.reshape(1, -1)
    t = pos.reshape(-1, 1) == neg.reshape(1, -1)
    return float((a.float() + 0.5 * t.float()).mean())


def matched(s, ev, cr):
    """Pool per-event comparisons against speed-matched cruise anchors only."""
    wins, tot, used = 0.0, 0, 0
    for i in torch.nonzero(ev).flatten():
        m = cr & (v0 - v0[i]).abs().le(TOL)
        if m.sum() < 3:
            continue
        used += 1
        c = s[m]
        wins += float((s[i] > c).float().sum() + 0.5 * (s[i] == c).float().sum())
        tot += int(m.sum())
    return (wins / tot if tot else float("nan")), used


def stratified(s, ev, cr, edges=(5, 8, 11, 14, 17, 21, 40)):
    num, den = 0.0, 0
    for lo, hi in zip(edges, edges[1:]):
        b = (v0 >= lo) & (v0 < hi)
        e, c = ev & b, cr & b
        if e.sum() >= 3 and c.sum() >= 5:
            num += auroc(s[e], s[c]) * int(e.sum())
            den += int(e.sum())
    return (num / den if den else float("nan")), den


out = {"tol_mps": TOL, "n_cruise": int(cruise.sum())}
for lbl, ev in LAB.items():
    row = {"n_events": int(ev.sum()),
           "median_v0_event": round(float(v0[ev].median()), 2),
           "median_v0_cruise": round(float(v0[cruise].median()), 2)}
    for k, s in sig.items():
        row[f"raw_{k}"] = round(auroc(s[ev], s[cruise]), 3)
        m, used = matched(s, ev, cruise)
        row[f"matched_{k}"] = round(m, 3)
        st, den = stratified(s, ev, cruise)
        row[f"strat_{k}"] = round(st, 3)
        row["n_events_matched"] = used
        row["n_events_stratified"] = den
    out[lbl] = row

print(json.dumps(out, indent=2))
with open(P.replace("_windows.pt", "_speedmatched.json"), "w") as f:
    json.dump(out, f, indent=2)
