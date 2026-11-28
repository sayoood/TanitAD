"""A/B the shipped v2.1 route label against the scene-adaptive v2.2, SAME windows.

PRE-REGISTERED READINGS (both committed in advance, 2026-08-03, before the run):

  OUTCOME A — v2.2 materially reduces the turn share and moves the freed mass into
    `road_following` / `gray_zone`. Reading: v2.1's `tight_transient` majority on
    20 s clips was a FALSE-TURN artefact of the disabled transience gate, and the
    strategic label that REF-C trains on today is mostly wrong. Action: retrain
    the strategic aux on v2.2 and re-run the T1 map-derived cross-check.

  OUTCOME B — the two agree on almost every window (`changed_from_v21` small).
    Reading: the turns v2.1 emits really are transient at 60 m as well as
    unmeasurable at 150 m, so the disabled gate was harmless HERE and the 65 %
    turn share is a property of this (slow, urban) corpus. Action: keep v2.1, and
    the corpus recommendation must instead address the SPEED MIX.

  Either way the map-derived NuRec labels (`strategic_gt.py`) remain the only
  non-kinematic arbiter, and neither outcome is evidence about them.

The script also reports the DISAGREEMENT CONDITIONED ON SPEED, because
`route_gate_speed_probe.json` already showed the turn rate is a strong function
of ego speed and a pooled delta would hide which regime moved.
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

import refb_labels as RL                                     # noqa: E402
from tanitad.data import parity as PARITY                    # noqa: E402

NAMES = {RL.ROUTE_LEFT: "left", RL.ROUTE_STRAIGHT: "straight",
         RL.ROUTE_RIGHT: "right", RL.ROUTE_UNKNOWN: "unknown"}
V_EDGES = [0.0, 1.0, 3.0, 6.0, 10.0, 1e9]
V_NAMES = ["v<1", "1-3", "3-6", "6-10", "10+"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=RL.LABEL_HORIZON)
    a = ap.parse_args()

    r21, r22 = Counter(), Counter()
    reason21, reason22 = Counter(), Counter()
    conf = Counter()
    by_v = {n: Counter() for n in V_NAMES}
    valid21 = valid22 = n = 0
    t0 = time.time()
    n_ep = 0
    splits = []
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
            if not ts:
                continue
            n_ep += 1
            for t in ts:
                x = RL.route_from_future_v22(P, int(t))
                v0 = float(P[t, 3])
                b = V_NAMES[int(np.digitize([v0], V_EDGES)[0]) - 1]
                n += 1
                r21[NAMES[x["v21_route"]]] += 1
                r22[NAMES[x["route"]]] += 1
                reason21[x["v21_reason"]] += 1
                reason22[x["reason"]] += 1
                conf[(NAMES[x["v21_route"]], NAMES[x["route"]])] += 1
                valid21 += int(x["v21_reason"] not in ("no_future", "no_arc",
                                                       "gray_zone"))
                valid22 += int(bool(x["valid"]))
                by_v[b]["n"] += 1
                by_v[b]["changed"] += int(bool(x["changed_from_v21"]))
                by_v[b]["turn21"] += int(x["v21_route"] in
                                         (RL.ROUTE_LEFT, RL.ROUTE_RIGHT))
                by_v[b]["turn22"] += int(x["route"] in
                                         (RL.ROUTE_LEFT, RL.ROUTE_RIGHT))
                by_v[b]["measurable22"] += int(bool(x["transience_measurable"]))

    def pct(x, y=n):
        return round(100.0 * x / y, 3) if y else None

    changed = sum(v for (i, j), v in conf.items() if i != j)
    out = {
        "tool": "route_v21_vs_v22_ab.py", "evidence_class": "MEASURED",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": splits, "n_episodes": n_ep, "n_windows": n,
        "stride": a.stride, "wall_s": round(time.time() - t0, 1),
        "PARITY_WARNING": (
            "no split above resolves to a registered parity key unless flagged "
            "so; a NON-PARITY sample is admissible evidence about a LABELER "
            "(a pure function of poses) but its percentages are not cross-arm "
            "comparable and must never be quoted as parity numbers."),
        "v21_class_share": {k: {"n": v, "pct": pct(v)} for k, v in r21.most_common()},
        "v22_class_share": {k: {"n": v, "pct": pct(v)} for k, v in r22.most_common()},
        "v21_reason_share": {k: {"n": v, "pct": pct(v)} for k, v in reason21.most_common()},
        "v22_reason_share": {k: {"n": v, "pct": pct(v)} for k, v in reason22.most_common()},
        "v21_valid_rate_pct": pct(valid21), "v22_valid_rate_pct": pct(valid22),
        "n_changed": changed, "pct_changed": pct(changed),
        "confusion_v21_to_v22": {f"{i}->{j}": v for (i, j), v in
                                 sorted(conf.items(), key=lambda kv: -kv[1])},
        "by_speed": {k: {
            "n": v["n"],
            "turn_rate_v21_pct": pct(v["turn21"], v["n"]),
            "turn_rate_v22_pct": pct(v["turn22"], v["n"]),
            "pct_changed": pct(v["changed"], v["n"]),
            "pct_transience_measurable_v22": pct(v["measurable22"], v["n"]),
        } for k, v in by_v.items() if v["n"]},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("n_episodes", "n_windows", "v21_class_share",
                       "v22_class_share", "v21_reason_share", "v22_reason_share",
                       "n_changed", "pct_changed", "by_speed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
