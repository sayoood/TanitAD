#!/usr/bin/env python3
"""The max-speed channel's TRAINED semantics on NavSim — the ego-future ORACLE — and its census
against the deployable input (the map posted limit). SPEC amendment A4.

⭐ WHY. refcv6 was trained with ``v_max`` = ``g_tac.goals.SPEED_BAND.v_hi_ms`` = the max of the
ego's OWN REALISED speed over [t0+2 s, t0+6 s] (run ``config.json``:
``max_speed_onehot_v6.provenance = "ego-future (oracle INPUT, ~1.7555 bits)"``;
``tanitad/data/v7_labels.py::oracle_max_speed``: *"deployment supplies a LIMIT the driver may not
reach. That is a TRAIN/DEPLOY MISMATCH"*). The containing-window one-hot therefore told the model
BOTH an upper and a lower bound on its own future max speed. This package feeds the map limit (the
PI's "map/nav set-speed service" stand-in), which is only an upper bound. This script measures,
without any model: how often the two disagree, and in which direction; and it writes the per-token
oracle that the PRIVILEGED diagnostic arm ``R6_VMAXORACLE`` reads (never a result, never a bar).

The oracle here is the TRAINING definition on NavSim's 2 Hz log: ``max |v|`` over the log frames
at t0 + 2.0, 2.5, …, 6.0 s (``ego_dynamic_state[:2]``, ego frame), 9 samples where the PhysicalAI
labels had 41 at 10 Hz (so it can only UNDER-read the max between samples — stated, not fixed).
A token whose log does not cover [t0+2.0, t0+6.0] s (end of log, or a gap > 1.0 s inside the
window) is ``no_future`` — the arm then gets ``valid = 0`` for it, counted.

    python code/vmax_oracle.py --split navtest --inputs <navtest_inputs.json.gz> \
        --logs-root D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test \
        --map raw/inputs/speed_limits_navtest.json --out raw/inputs/vmax_oracle_navtest.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WIN_LO_S, WIN_HI_S = 2.0, 6.0         # tanitad/data/tactical_goals.py BAND_LO_S, BAND_HI_S
TOL_S = 0.05                          # 2 Hz timestamps jitter by a few ms (MEASURED 499,379-500,079 us)
MAX_GAP_S = 1.0


def oracle_from_frames(t0_us: int, frames: list) -> dict:
    """``frames``: the log's frame dicts (``timestamp`` us, ``ego_dynamic_state`` [vx, vy, ax, ay]).
    Returns ``{status, v_hi_ms, n_samples, offsets_s}`` for the window [t0+2, t0+6] s."""
    pts = []
    for f in frames:
        dt = (int(f["timestamp"]) - int(t0_us)) / 1e6
        if WIN_LO_S - TOL_S <= dt <= WIN_HI_S + TOL_S:
            vx, vy = float(f["ego_dynamic_state"][0]), float(f["ego_dynamic_state"][1])
            pts.append((dt, math.hypot(vx, vy)))
    pts.sort()
    if not pts:
        return {"status": "no_future", "reason": "no log frame in the window", "n_samples": 0,
                "why": "ORACLE unavailable: no log frame in [t0+2, t0+6] s"}
    offs = [p[0] for p in pts]
    gaps = [b - a for a, b in zip(offs, offs[1:])]
    if offs[0] > WIN_LO_S + TOL_S or offs[-1] < WIN_HI_S - TOL_S or (gaps and max(gaps) > MAX_GAP_S + TOL_S):
        return {"status": "no_future", "reason": "window not covered", "n_samples": len(pts),
                "why": "ORACLE unavailable: [t0+2, t0+6] s not covered by the log",
                "first_s": round(offs[0], 3), "last_s": round(offs[-1], 3),
                "max_gap_s": round(max(gaps), 3) if gaps else None}
    return {"status": "limit", "speed_limit_mps": max(p[1] for p in pts), "n_samples": len(pts),
            "first_s": round(offs[0], 3), "last_s": round(offs[-1], 3),
            "why": "ORACLE: the human's own realised max |v| over [t0+2, t0+6] s (PRIVILEGED)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--logs-root", required=True)
    ap.add_argument("--map", required=True, help="this package's speed_limits_<split>.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    import frames416 as F4                                               # E2's log loader
    from tanitad.refs.refcv6_max_speed import speed_max_bin, SPEED_MAX_STEPS_KMH_V6
    op = gzip.open if a.inputs.endswith(".gz") else open
    with op(a.inputs, "rt", encoding="utf-8") as fh:
        toks = json.load(fh)["tokens"]
    mp = json.load(open(a.map, encoding="utf-8"))
    if mp.get("split") != a.split:
        sys.exit(f"⛔ map file split {mp.get('split')} != {a.split}")
    by_log = {}
    for t, r in toks.items():
        if r.get("stage", 1) == 1:
            by_log.setdefault(r["log_name"], []).append(t)
    out = {}
    for i, (ln, tl) in enumerate(sorted(by_log.items())):
        fr = F4.E2BF.load(os.path.join(a.logs_root, f"{ln}.pkl"))
        for t in tl:
            rec = oracle_from_frames(int(toks[t]["timestamps_us"][-1]), fr)
            if rec["status"] == "limit":
                b, over = speed_max_bin(rec["speed_limit_mps"])
                rec["bin_kmh"] = SPEED_MAX_STEPS_KMH_V6[b]
                rec["over_ceiling"] = bool(over)
            out[t] = rec
        del fr
        if (i + 1) % 20 == 0:
            print(f"[vmax_oracle] {i + 1}/{len(by_log)} logs", flush=True)
    # ---- census: the oracle's own distribution, and map-vs-oracle -------------------------
    ok = [t for t, r in out.items() if r["status"] == "limit"]
    dist = Counter(out[t]["bin_kmh"] for t in ok)
    conf, dirn = Counter(), Counter()
    for t in ok:
        m = mp["tokens"].get(t)
        if not m or m.get("status") != "limit":
            conf[("unknown", out[t]["bin_kmh"])] += 1
            continue
        mb, _ = speed_max_bin(float(m["speed_limit_mps"]))
        mk = SPEED_MAX_STEPS_KMH_V6[mb]
        conf[(mk, out[t]["bin_kmh"])] += 1
        dirn["map_above_oracle" if mk > out[t]["bin_kmh"] else
             ("equal" if mk == out[t]["bin_kmh"] else "map_below_oracle")] += 1
    n_map = sum(dirn.values())
    census = {
        "n_tokens": len(out), "n_oracle_ok": len(ok),
        "n_no_future": len(out) - len(ok),
        "oracle_bin_share": {str(k): round(v / max(len(ok), 1), 4) for k, v in sorted(dist.items())},
        "training_bin_share_reference": {"30": 0.3808, "50": 0.3484, "100": 0.2218, "120": 0.0490,
                                         "_source": "tanitad/refs/refcv6_max_speed.py docstring "
                                                    "(v8 train blob, n = 4,572, MEASURED there)"},
        "map_vs_oracle_on_tokens_with_both": {
            "n": n_map, **{k: v for k, v in dirn.items()},
            **{f"frac_{k}": round(v / max(n_map, 1), 4) for k, v in dirn.items()}},
        "confusion_map_bin__oracle_bin": {f"{k[0]}->{k[1]}": v for k, v in sorted(
            conf.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))},
        "_read": ("'map_above_oracle' = the deployable input tells the model a HIGHER band than the "
                  "human actually drove in [t0+2, t0+6] s — the direction the containing-window "
                  "one-hot never showed it in training (there, the realised max is INSIDE the band)."),
    }
    doc = {"split": a.split, "_what": __doc__.split("\n\n")[0], "window_s": [WIN_LO_S, WIN_HI_S],
           "sampling": "2 Hz log frames (9 samples) — can only UNDER-read the 10 Hz training max",
           "inputs": os.path.abspath(a.inputs), "logs_root": a.logs_root,
           "map": os.path.abspath(a.map), "census": census, "tokens": out}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(json.dumps(census, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
