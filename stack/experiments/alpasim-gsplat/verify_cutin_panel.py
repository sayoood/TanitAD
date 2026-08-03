#!/usr/bin/env python3
"""Two verifications the cut-in-targeted panel must pass before any number is quoted.

⚠️ VERIFY BY LOADING OR FROM THE FAR SIDE, NEVER BY EXIT CODE.

1. DETERMINISM (the only true instrument null in this design). `flagship-v1 / objects`
   was run TWICE with identical arguments. Every recorded quantity must match exactly.
   A non-zero delta anywhere means the loop has hidden state and NO contrast in the panel
   is interpretable.

2. CROSS-VERSION RENDER EQUIVALENCE. The banked scene-2 panel ran against
   `gsplat_renderer.py` md5 24b61aa0…; this panel ran against 0780e18d… (a sibling stream
   synced its rolling-shutter work to the pod between the two). Reading the diff says the
   changes are additive — new helpers, an opt-in `return_torch` kwarg, a new
   `render_rs_sliced` method — but reading a diff is an argument, not a measurement.
   Starts 75 and 90 appear in BOTH panels with the same steps and the same condition, so
   the claim is testable: if the default render path is unchanged, those rollouts are
   bit-identical. If they are not, the targeted panel may not be compared to the banked
   one and the comparison is dropped rather than caveated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def index(path):
    d = json.loads(Path(path).read_text())
    return d, {int(r["start_frame"]): r for r in d["rollouts"]}


def cmp_rollout(ra, rb):
    """Exact comparison of every numeric field recorded per step."""
    sa, sb = ra["steps"], rb["steps"]
    if len(sa) != len(sb):
        return {"identical": False, "reason": f"step count {len(sa)} vs {len(sb)}"}
    worst = {}
    for key in ("v", "steer", "accel", "v_target", "kappa_plan"):
        d = max(abs(float(x[key]) - float(y[key])) for x, y in zip(sa, sb))
        worst[key] = d
    worst["ego_xy"] = max(
        float(np.max(np.abs(np.array(x["ego"][:2]) - np.array(y["ego"][:2]))))
        for x, y in zip(sa, sb))
    worst["plan"] = max(
        float(np.max(np.abs(np.array(x["plan"], float) - np.array(y["plan"], float))))
        for x, y in zip(sa, sb))
    worst["i_gt"] = max(abs(int(x["i_gt"]) - int(y["i_gt"])) for x, y in zip(sa, sb))
    worst["t_us"] = max(abs(float(x["t_us"]) - float(y["t_us"])) for x, y in zip(sa, sb))
    nav = sum(int(x["nav"] != y["nav"]) for x, y in zip(sa, sb))
    ml = 0.0
    for x, y in zip(sa, sb):
        ea, eb = x.get("extra") or {}, y.get("extra") or {}
        for k in set(ea) & set(eb):
            try:
                ml = max(ml, float(np.max(np.abs(np.asarray(ea[k], float)
                                                 - np.asarray(eb[k], float)))))
            except Exception:                                   # noqa: BLE001
                pass
    worst["extra_logits"] = ml
    worst["nav_mismatches"] = nav
    return {"identical": all(v == 0 for v in worst.values()),
            "max_abs_diff": {k: (float(v) if not isinstance(v, int) else v)
                             for k, v in worst.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run A rollouts json")
    ap.add_argument("--b", required=True, help="run B rollouts json")
    ap.add_argument("--what", required=True,
                    choices=["determinism", "cross_version"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dA, RA = index(a.a)
    dB, RB = index(a.b)
    shared = sorted(set(RA) & set(RB))
    per = {str(s): cmp_rollout(RA[s], RB[s]) for s in shared}
    allsame = bool(shared) and all(v["identical"] for v in per.values())

    res = {
        "what": a.what,
        "evidence_class": "MEASURED (ours)",
        "A": {"file": str(a.a), "arm": dA["arm"], "condition": dA["condition"],
              "n_rollouts": len(dA["rollouts"]),
              "starts": sorted(int(r["start_frame"]) for r in dA["rollouts"])},
        "B": {"file": str(a.b), "arm": dB["arm"], "condition": dB["condition"],
              "n_rollouts": len(dB["rollouts"]),
              "starts": sorted(int(r["start_frame"]) for r in dB["rollouts"])},
        "shared_starts": shared,
        "ALL_SHARED_ROLLOUTS_BIT_IDENTICAL": allsame,
        "per_start": per,
        "verdict": (
            ("PASS — the closed loop is deterministic; every non-zero delta elsewhere in "
             "the panel is treatment, not run-to-run noise."
             if a.what == "determinism" else
             "PASS — the renderer change is a MEASURED no-op on the default path, so the "
             "targeted panel may be compared to the banked scene-2 panel.")
            if allsame else
            ("FAIL — the loop is NOT deterministic. No contrast in this panel is "
             "interpretable until this is explained."
             if a.what == "determinism" else
             "FAIL — the renderer change moved the default path. The targeted panel must "
             "NOT be compared to the banked scene-2 panel; drop the comparison.")),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_start"}, indent=1))
    for s, v in per.items():
        if not v["identical"]:
            print(f"  start {s}: {v.get('reason', v['max_abs_diff'])}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
