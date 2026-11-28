"""P4 — HOW MUCH OF THE CORPUS'S `follow` SUPERVISION IS THE UNKNOWN SENTINEL, BY SPEED.

`nav_command_v21` maps BOTH `ROUTE_STRAIGHT` and `ROUTE_UNKNOWN` onto `NAV_FOLLOW`, so a
`follow` token at the model input can be a route STATEMENT or a CONFESSION and the model
cannot tell them apart. `nav_command_v21_ex` exposes the bit (`unknown_sentinel`); this
probe counts it per SPEED STRATUM, on the same strata and the same window construction as
`route_gate_speed_probe.py`, so the two are directly comparable.

THE QUESTION IT ANSWERS. The tactical lossy rate is strongly speed-dependent (MEASURED:
38.2 % at 1-3 m/s -> 1.8 % at 10-15 m/s). If the nav collapse tracks the SAME axis, one
mechanism — a curvature estimator whose variance grows as 1/v — degrades the tactical and
the strategic label together, and a single fix reaches both. If it runs the OTHER way, the
strategic level is unlearnable in exactly the regime where the tactical level is fine, and
they need separate fixes. That is a discriminating measurement, not a description.

⚠️ PARITY. The dev box holds `physicalai-train-14231cd29c74` and
`physicalai-val-bb543bdf7836`, which `tanitad.data.parity.corpus_key_of` resolves to
**None** — they are NOT the canonical `physicalai-train-e438721ae894`. These numbers are
admissible as evidence about the LABELLER (a pure function of poses) and about the
MECHANISM; they are NOT cross-arm comparable and no percentage here is a parity number.
Nothing is re-selected: the probe only reads.
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

V_EDGES = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 1e9]
V_NAMES = ["v<1", "1-3", "3-6", "6-10", "10-15", "15+"]
NAV_NAMES = ("follow", "left", "right", "straight")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=0)
    a = ap.parse_args()

    strata = {n: Counter() for n in V_NAMES}
    reason_follow = {n: Counter() for n in V_NAMES}
    t0, n_ep = time.time(), 0
    for cd in a.cache_dir:
        for f in sorted(Path(cd).glob("ep_*.pt")):
            if a.max_episodes and n_ep >= a.max_episodes:
                break
            P = torch.load(f, map_location="cpu",
                           weights_only=False)["poses"].to(torch.float32)
            T = int(P.shape[0])
            ts = list(range(0, T - 1, a.stride))
            if not ts:
                continue
            n_ep += 1
            v0 = P[torch.tensor(ts), 3].numpy()
            b = np.digitize(v0, V_EDGES) - 1
            for i, t in enumerate(ts):
                d = RL.nav_command_v21_ex(P, int(t))
                s = strata[V_NAMES[int(b[i])]]
                s["n"] += 1
                s[f"nav_{NAV_NAMES[d['nav']]}"] += 1
                if d["nav"] == RL.NAV_FOLLOW:
                    s["follow"] += 1
                    if d["unknown_sentinel"]:
                        s["follow_unknown_sentinel"] += 1
                    reason_follow[V_NAMES[int(b[i])]][d["reason"]] += 1
                if not d["valid"]:
                    s["invalid"] += 1

    out = {"tool": "nav_sentinel_by_speed.py", "evidence_class": "MEASURED",
           "cache_dirs": a.cache_dir, "stride": a.stride, "n_episodes": n_ep,
           "wall_s": round(time.time() - t0, 1),
           "parity_note": "NOT the canonical physicalai-train-e438721ae894; "
                          "corpus_key_of resolves these to None. Labeller evidence "
                          "only — never a cross-arm number.",
           "speed_strata": {}, "follow_reason_by_speed": {}}
    tot = sum(strata[n]["n"] for n in V_NAMES)
    for n in V_NAMES:
        s = strata[n]
        if not s["n"]:
            continue
        fol = s["follow"]
        out["speed_strata"][n] = {
            "n": s["n"],
            "pct_of_corpus": round(100.0 * s["n"] / max(tot, 1), 3),
            "follow_pct_of_stratum": round(100.0 * fol / s["n"], 3),
            "follow_unknown_sentinel_n": s["follow_unknown_sentinel"],
            "SENTINEL_pct_of_follow": (None if not fol else
                                       round(100.0 * s["follow_unknown_sentinel"]
                                             / fol, 3)),
            "SENTINEL_pct_of_stratum": round(
                100.0 * s["follow_unknown_sentinel"] / s["n"], 3),
            "turn_pct_of_stratum": round(
                100.0 * (s["nav_left"] + s["nav_right"]) / s["n"], 3),
            "invalid_pct": round(100.0 * s["invalid"] / s["n"], 3)}
        out["follow_reason_by_speed"][n] = {
            k: {"n": v, "pct": round(100.0 * v / max(fol, 1), 2)}
            for k, v in reason_follow[n].most_common()}
    allf = sum(strata[n]["follow"] for n in V_NAMES)
    alls = sum(strata[n]["follow_unknown_sentinel"] for n in V_NAMES)
    out["corpus_total"] = {
        "n_windows": tot, "n_follow": allf, "n_follow_unknown_sentinel": alls,
        "SENTINEL_pct_of_follow": (None if not allf
                                   else round(100.0 * alls / allf, 3))}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["speed_strata"], indent=1))
    print(json.dumps(out["corpus_total"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
