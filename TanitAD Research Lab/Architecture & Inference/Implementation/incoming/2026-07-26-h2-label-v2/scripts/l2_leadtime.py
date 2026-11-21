"""H2 L2 — LEAD TIME. Does the trigger fire BEFORE the ego starts braking, or after?

This is the discriminating measurement for the one caveat the CONFIRM run left open: adjusting for
the ego's braking state at t drops the lift from 2.41x to 1.35x [0.82, 2.05]. That is either

  (a) OVER-ADJUSTMENT for a MEDIATOR -- the driver begins reacting before the conflict geometry
      peaks, so "already braking at t" lies ON the causal path and conditioning on it blocks the
      effect being measured;  or
  (b) adjustment for a genuine CONFOUNDER -- dense slow traffic causes both.

They make opposite predictions about ORDER. Under (a) the trigger onset should systematically
PRECEDE the brake onset. Under (b) there is no reason for it to.

Descriptive. Post-verdict. Cannot move the verdict.

usage:  python l2_leadtime.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from l2_label import CONFIRM_CHUNKS, trigger_l2                       # noqa: E402

TAB = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\l2tab")
OUT = os.path.abspath(os.path.join(HERE, ".."))
BRAKE = -2.0                     # same magnitude as R2, applied to the TRAILING mean = "braking now"

dev = json.load(open(os.path.join(OUT, "l2_dev.json")))
TAU, RESOLVABLE = dev["tau_star"], dev["A2_decision_exclude_unresolvable"]
D = pd.concat([pd.read_parquet(os.path.join(TAB, f"l2_{c}.parquet")) for c in CONFIRM_CHUNKS],
              ignore_index=True)
D["trig"] = trigger_l2(D, TAU, resolvable=RESOLVABLE)
D["braking_now"] = D.alon_pre.to_numpy() <= BRAKE

leads, n_no_brake = [], 0
for cid, sub in D.groupby("clip_id", sort=False):
    s = sub.sort_values("gi")
    t = s.trig.to_numpy()
    b = s.braking_now.to_numpy()
    if not t.any():
        continue
    k_t = int(np.argmax(t))
    # first brake onset at or after the trigger onset, within the 4 s label horizon
    w = b[k_t:k_t + 41]
    if not w.any():
        n_no_brake += 1
        continue
    leads.append(int(np.argmax(w)) * 0.1)          # seconds from trigger onset to brake onset

leads = np.asarray(leads)
res = {
    "definition": "per CONFIRM clip: t_trigger = first L2_trigger frame; t_brake = first frame with "
                  f"trailing-0.5 s alon <= {BRAKE} m/s^2 in [t_trigger, t_trigger + 4 s]",
    "tau": TAU,
    "n_trigger_positive_clips": int(len(leads) + n_no_brake),
    "n_with_brake_in_horizon": int(len(leads)),
    "n_without_brake_in_horizon": int(n_no_brake),
    "share_already_braking_at_trigger_onset (lead = 0.0 s)": round(float((leads == 0).mean()), 4)
    if len(leads) else None,
    "share_trigger_STRICTLY_precedes_brake": round(float((leads > 0).mean()), 4) if len(leads) else None,
    "lead_seconds_p25_p50_p75": [round(float(np.percentile(leads, q)), 2) for q in (25, 50, 75)]
    if len(leads) else None,
    "mean_lead_s": round(float(leads.mean()), 3) if len(leads) else None,
}
print(json.dumps(res, indent=2))

# the same statistic for a matched control: clips WITHOUT any trigger, anchored at a random frame
rng = np.random.default_rng(0)
ctrl = []
for cid, sub in D.groupby("clip_id", sort=False):
    s = sub.sort_values("gi")
    if s.trig.to_numpy().any() or len(s) < 60:
        continue
    k = int(rng.integers(0, len(s) - 41))
    w = s.braking_now.to_numpy()[k:k + 41]
    if w.any():
        ctrl.append(int(np.argmax(w)) * 0.1)
ctrl = np.asarray(ctrl)
res["control_no_trigger_clips"] = {
    "n": int(len(ctrl)),
    "share_braking_at_anchor": round(float((ctrl == 0).mean()), 4) if len(ctrl) else None,
    "P(any brake onset in the 4 s window)": round(
        float(len(ctrl) / max(sum(1 for _, s in D.groupby('clip_id') if not s.trig.any()), 1)), 4)}
print("\ncontrol:", json.dumps(res["control_no_trigger_clips"], indent=2))

json.dump(res, open(os.path.join(OUT, "l2_leadtime.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'l2_leadtime.json')}")
