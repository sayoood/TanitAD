"""WHY THE ROUTE / TURN LABEL DEPENDS ON THE CORPUS'S SPEED DISTRIBUTION.

The tactical turn gate and the strategic route gate are both CURVATURE tests
(``kappa = dyaw / ds``, 1/m). Curvature is speed-INVARIANT for a given road
geometry — that is the whole reason v2 moved to it. But the ESTIMATOR is not:
at low speed ``ds`` per step collapses toward the ``MIN_ARC_M`` floor while the
yaw noise floor does not, so ``kappa`` is measured with a variance that grows as
1/v. A corpus whose speed distribution shifts down therefore reports MORE turns
for the same roads.

This probe measures that dependence directly, and simultaneously answers the
question that matters for the corpus recommendation: *does the episode selection
change the label distribution?* It stratifies the SAME windows by ego speed and
reports the turn rate, the route reason histogram and the lossy rate per stratum.

It also reports the observed ARC per window against the two arc constants
``route_from_future_v21`` gates on, because on 20 s clips one of them is
unreachable and that silently disables half the decision rule.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve()
while _REPO.name != "TanitAD" and _REPO.parent != _REPO:
    _REPO = _REPO.parent
for p in (str(_REPO / "stack"), str(_REPO / "stack" / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import refb_labels as RL                                   # noqa: E402
from tanitad.refs import refc_tactical as TAC              # noqa: E402

V_EDGES = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 1e9]
V_NAMES = ["v<1", "1-3", "3-6", "6-10", "10-15", "15+"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=RL.LABEL_HORIZON)
    a = ap.parse_args()

    strata = {n: Counter() for n in V_NAMES}
    reason_by_v = {n: Counter() for n in V_NAMES}
    arcs, v0s, kaps = [], [], []
    t0 = time.time()
    n_ep = 0
    for cd in a.cache_dir:
        for f in sorted(Path(cd).glob("ep_*.pt")):
            P = torch.load(f, map_location="cpu",
                           weights_only=False)["poses"].to(torch.float32)
            T = int(P.shape[0])
            ts = list(range(0, T - a.horizon, a.stride))
            if not ts:
                continue
            n_ep += 1
            idx = torch.tensor(ts)
            pl = P[idx]
            fut = torch.stack([P[t + 1: t + 1 + a.horizon] for t in ts])
            lat, lon = TAC.window_factored_labels_v2(pl, fut, a.horizon)
            dyaw = RL.wrap_to_pi(fut[:, a.horizon - 1, 2] - pl[:, 2])
            seg = (fut[:, 1:a.horizon, :2] - fut[:, :a.horizon - 1, :2]).norm(dim=-1)
            arc2s = seg.sum(1) + (fut[:, 0, :2] - pl[:, :2]).norm(dim=-1)
            kap = (dyaw / arc2s.clamp_min(RL.MIN_ARC_M)).numpy()
            v0 = pl[:, 3].numpy()
            b = np.digitize(v0, V_EDGES) - 1
            for i, t in enumerate(ts):
                s = strata[V_NAMES[int(b[i])]]
                s["n"] += 1
                s["turn"] += int(lat[i].item() != TAC.LAT_LANE_KEEP)
                s["live_lon"] += int(lon[i].item() != TAC.LON_STEADY)
                s["lossy"] += int(lat[i].item() != TAC.LAT_LANE_KEEP
                                  and lon[i].item() != TAC.LON_STEADY)
                s["arc2s_sum"] += float(arc2s[i])
                r = RL.route_from_future_v21(P, int(t))
                reason_by_v[V_NAMES[int(b[i])]][r["reason"]] += 1
                arcs.append(float(r["arc_m"]))
                v0s.append(float(v0[i]))
                kaps.append(abs(float(kap[i])))

    arcs_a, v0_a, kap_a = np.array(arcs), np.array(v0s), np.array(kaps)
    tot = int(sum(s["n"] for s in strata.values()))
    out = {
        "tool": "route_gate_speed_probe.py", "evidence_class": "MEASURED",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_dirs": list(a.cache_dir), "stride": a.stride,
        "n_episodes": n_ep, "n_windows": tot, "wall_s": round(time.time() - t0, 1),
        "speed_strata": {
            k: {
                "n": s["n"], "pct_of_corpus": round(100 * s["n"] / max(tot, 1), 3),
                "turn_rate_pct": round(100 * s["turn"] / max(s["n"], 1), 3),
                "live_longitudinal_pct": round(100 * s["live_lon"] / max(s["n"], 1), 3),
                "LOSSY_pct": round(100 * s["lossy"] / max(s["n"], 1), 3),
                "mean_2s_arc_m": round(s["arc2s_sum"] / max(s["n"], 1), 2),
            } for k, s in strata.items()},
        "route_reason_by_speed": {
            k: {r: {"n": n, "pct": round(100 * n / max(sum(c.values()), 1), 2)}
                for r, n in c.most_common()} for k, c in reason_by_v.items()},
        "arc_vs_gates": {
            "MIN_ARC_ROUTE_M": RL.MIN_ARC_ROUTE_M,
            "TRANSIENCE_MIN_ARC_M": RL.TRANSIENCE_MIN_ARC_M,
            "CONC_ARC_M": RL.CONC_ARC_M,
            "median_observed_arc_m": round(float(np.median(arcs_a)), 2),
            "p90_observed_arc_m": round(float(np.percentile(arcs_a, 90)), 2),
            "pct_windows_below_MIN_ARC_ROUTE_M":
                round(100 * float((arcs_a < RL.MIN_ARC_ROUTE_M).mean()), 3),
            "pct_windows_below_TRANSIENCE_MIN_ARC_M":
                round(100 * float((arcs_a < RL.TRANSIENCE_MIN_ARC_M).mean()), 3),
            "VERDICT": (
                "the transience half of the v2.1 turn rule is DISABLED on this "
                "corpus: `transient` is forced True whenever arc < "
                "TRANSIENCE_MIN_ARC_M, so on the fraction printed above the rule "
                "degenerates to the tightness test ALONE — exactly the "
                "false-turn failure mode the concentration gate was added to "
                "prevent (refb_labels v2 header)."),
        },
        "kappa_vs_speed": {
            "note": ("|window-mean curvature| vs ego speed. The turn gate is "
                     f"|kappa| >= {round(RL.CURV_TURN_MAN_PER_M, 6)} 1/m and it is "
                     "the SAME number at every speed, so any monotone trend here "
                     "is estimator variance, not road geometry."),
            "per_stratum": {},
        },
    }
    b = np.digitize(v0_a, V_EDGES) - 1
    for i, nm in enumerate(V_NAMES):
        m = b == i
        if m.sum():
            out["kappa_vs_speed"]["per_stratum"][nm] = {
                "n": int(m.sum()),
                "median_abs_kappa": round(float(np.median(kap_a[m])), 6),
                "p90_abs_kappa": round(float(np.percentile(kap_a[m], 90)), 6),
                "pct_above_turn_gate": round(
                    100 * float((kap_a[m] >= RL.CURV_TURN_MAN_PER_M).mean()), 3)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("n_episodes", "n_windows", "speed_strata",
                       "arc_vs_gates", "kappa_vs_speed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
