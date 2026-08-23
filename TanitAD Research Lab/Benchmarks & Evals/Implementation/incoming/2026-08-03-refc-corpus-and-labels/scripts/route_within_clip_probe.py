"""IS THE STRATEGIC LABEL A PER-WINDOW DECISION, OR A PER-CLIP TAG?

The v2.1 route label at ``t`` is decided by ``peak_kappa`` — the MAXIMUM smoothed
curvature ANYWHERE in the next ~20 s. On a 20 s clip that lookahead is the whole
remaining clip, so if the ego turns at all, every window before the turn inherits
the same class. The label is then not a time-varying strategic decision at all: it
is a CLIP-LEVEL TAG, repeated once per window, and a head trained on it can score
well while carrying no information about *when* the manoeuvre is due.

This is the failure mode the v3 DISTANCE-TO-MANEUVER tokens exist to fix
(``refb_labels`` v3 header: *"`roundabout` is a shape; `roundabout, exit in 40 m`
is an instruction"*), and it is measurable directly:

  * ``pct_episodes_with_a_constant_route``   — the label never changes in the clip
  * ``mean_distinct_classes_per_episode``    — 1.0 means a pure clip tag
  * ``mean_switches_per_episode``            — how often it changes at all
  * the same three for the FACTORED tactical label, as a contrast: a tactical
    label SHOULD vary within a clip, so if it does and the route does not, the
    difference is a property of the route derivation and not of the corpus.
  * ``dist_to_turn_m`` at the windows labelled a turn — the quantity a planner
    needs and the label does not carry.

Read-only, CPU-only, poses only.
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

import refb_labels as RL                                    # noqa: E402
from tanitad.data import parity as PARITY                   # noqa: E402
from tanitad.refs import refc_tactical as TAC               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=RL.LABEL_HORIZON)
    a = ap.parse_args()

    n_ep = 0
    route_distinct, route_switch, route_const = [], [], 0
    man_distinct, man_switch, man_const = [], [], 0
    lat_distinct, lon_distinct = [], []
    dists, band = [], Counter()
    splits = []
    t0 = time.time()
    for cd in a.cache_dir:
        d = Path(cd)
        splits.append({"cache_dir": str(d),
                       "resolved_corpus_key": PARITY.corpus_key_of(d),
                       "is_registered_parity_corpus":
                           PARITY.corpus_key_of(d) is not None})
        for f in sorted(d.glob("ep_*.pt")):
            P = torch.load(f, map_location="cpu",
                           weights_only=False)["poses"].to(torch.float32)
            T = int(P.shape[0])
            ts = list(range(0, T - a.horizon, a.stride))
            if len(ts) < 2:
                continue
            n_ep += 1
            rr = [RL.route_from_future_v21(P, int(t))["route"] for t in ts]
            idx = torch.tensor(ts)
            pl, fut = P[idx], torch.stack([P[t + 1: t + 1 + a.horizon] for t in ts])
            m5 = RL.window_maneuver_labels_v2(pl, fut, a.horizon).tolist()
            lat, lon = TAC.window_factored_labels_v2(pl, fut, a.horizon)
            route_distinct.append(len(set(rr)))
            route_switch.append(sum(rr[i] != rr[i - 1] for i in range(1, len(rr))))
            route_const += int(len(set(rr)) == 1)
            man_distinct.append(len(set(m5)))
            man_switch.append(sum(m5[i] != m5[i - 1] for i in range(1, len(m5))))
            man_const += int(len(set(m5)) == 1)
            lat_distinct.append(len(set(lat.tolist())))
            lon_distinct.append(len(set(lon.tolist())))
            # distance to the FIRST junction-scale turn ahead, per turn window
            for t, r in zip(ts, rr):
                if r in (RL.ROUTE_LEFT, RL.ROUTE_RIGHT):
                    seg = P[t:]
                    ds = (seg[1:, :2] - seg[:-1, :2]).norm(dim=-1)
                    dyaw = RL.wrap_to_pi(seg[1:, 2] - seg[:-1, 2])
                    kap = torch.where(ds >= RL.MIN_ARC_M,
                                      dyaw / ds.clamp_min(RL.MIN_ARC_M),
                                      torch.zeros_like(ds))
                    ks = RL._moving_avg(kap, RL._arc_smooth_k(ds))
                    if ks.numel() == 0:
                        continue
                    j = int(ks.abs().argmax())
                    dm = float(ds[:j].sum())
                    dists.append(dm)
                    band[RL.dist_band(dm, observed_arc_m=float(ds.sum()))] += 1

    def stat(v):
        v = np.array(v, dtype=np.float64)
        return {"n": int(len(v)), "mean": round(float(v.mean()), 4),
                "median": round(float(np.median(v)), 3),
                "p90": round(float(np.percentile(v, 90)), 3)} if len(v) else {}

    D = np.array(dists) if dists else np.zeros(0)
    out = {
        "tool": "route_within_clip_probe.py", "evidence_class": "MEASURED",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": splits, "n_episodes": n_ep, "stride": a.stride,
        "wall_s": round(time.time() - t0, 1),
        "PARITY_WARNING": ("percentages here are NOT parity numbers unless a split "
                           "above resolves to a registered key"),
        "ROUTE_label_within_clip": {
            "pct_episodes_with_a_CONSTANT_route":
                round(100.0 * route_const / max(n_ep, 1), 3),
            "distinct_classes_per_episode": stat(route_distinct),
            "switches_per_episode": stat(route_switch),
        },
        "TACTICAL_man5_within_clip": {
            "pct_episodes_with_a_CONSTANT_man5":
                round(100.0 * man_const / max(n_ep, 1), 3),
            "distinct_classes_per_episode": stat(man_distinct),
            "switches_per_episode": stat(man_switch),
            "lat_distinct_per_episode": stat(lat_distinct),
            "lon_distinct_per_episode": stat(lon_distinct),
        },
        "DISTANCE_TO_MANEUVER": {
            "n_turn_windows": int(len(D)),
            "median_m": round(float(np.median(D)), 2) if len(D) else None,
            "p90_m": round(float(np.percentile(D, 90)), 2) if len(D) else None,
            "pct_beyond_50m": round(100.0 * float((D > 50).mean()), 3) if len(D) else None,
            "band_share": {k: {"n": v, "pct": round(100.0 * v / max(len(D), 1), 3)}
                           for k, v in band.most_common()},
            "why_it_matters": (
                "these windows all carry the SAME route token. A planner needs "
                "`turn_left in 12 m` and `turn_left in 90 m` to be different "
                "instructions; the shipped 3-class label cannot express the "
                "difference, and refb_labels.DIST_BAND_TOKENS (width 8) is minted "
                "by route_from_future_v3 but NOT enrolled in lake/vocab.py."),
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("n_episodes", "ROUTE_label_within_clip",
                       "TACTICAL_man5_within_clip", "DISTANCE_TO_MANEUVER")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
