"""PREREG_V8F_MINI Amendment 4: the verdict over the full-corpus M1 / M2b runs (box 2), exactly as committed before any data.

  python v8f_mini_full_verdict.py --eval-dir <dir with eval_<tag>.npz/.json, compare.json, floors_report.json> --runs-dir <dir with
         <arm>_s<k>/summary.json> --out verdict.json

Reads (never re-estimates what `v8f_mini_eval.py compare` already decided):
  * PROVENANCE: every run trained on the SAME 4,512 clips (one train_sha256), withheld the RA-1 holdout (excluded_sha256 = HOLDOUT_SHA),
    and finished (summary "done"). Anything else REFUSES: a verdict over mixed data is not a verdict.
  * H-MINI-K-1, per arm, from compare.json (ADE 6 s arm - P10-yaw-4512, paired, separated at BOTH seeds, |delta| > 2F). Amendment 4: no
    arm is SELECTED on eval-141 (it scores K); both arms are read, K-1 passes iff at least one arm passes, and the 2-test multiplicity
    is stated in the output.
  * H-MINI-WM-1 (new in Amendment 4): M2b - M1, ADE 6 s, paired per seed (s0 with s0, s1 with s1), separated at BOTH seeds, mean < 0,
    |delta| > 2F with F = max(F_M1, F_M2b). The estimator is the programme's (`taniteval.ci`), not re-implemented.
  * the four families + consistency + the "acts" check, per run, copied from each eval_<tag>.json (and P10's from floors_report.json):
    STRATEGIC is un-paused for v8F-mini (PI ruling of 2026-10-17), so it is read, not paused.
Estimator, every interval: the paired episode-cluster bootstrap over the 141 eval episodes, B = 2,000. It answers ANOTHER DRAW OF
EPISODES only; training-seed variance is answered by F alone, and with 2 seeds F is itself one draw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
ARMS, SEEDS = ("M1", "M2b"), (0, 1)
N_TRAIN = 4512
B = 2000


def load_runs(runs_dir):
    S = {}
    for arm in ARMS:
        for s in SEEDS:
            p = os.path.join(runs_dir, f"{arm}_s{s}", "summary.json")
            if not os.path.exists(p):
                raise SystemExit(f"REFUSED: {p} is missing (the run did not finish)")
            S[(arm, s)] = json.load(open(p, encoding="utf-8"))
    bad = []
    if {d["train_sha256"] for d in S.values()} != {S[("M1", 0)]["train_sha256"]}:
        bad.append("the runs did not train on one clip list")
    if {d.get("excluded_sha256") for d in S.values()} != {HOLDOUT_SHA}:
        bad.append(f"a run did not withhold the RA-1 holdout: {sorted({str(d.get('excluded_sha256')) for d in S.values()})}")
    if {d["n_clips"] for d in S.values()} != {N_TRAIN}:
        bad.append(f"n_clips {sorted({d['n_clips'] for d in S.values()})} != {N_TRAIN}")
    if not all(d.get("done") is True for d in S.values()):
        bad.append("a run is not done")
    if len({d["steps"] for d in S.values()}) != 1:
        bad.append(f"the runs trained for different step counts: {sorted({d['steps'] for d in S.values()})}")
    for (arm, s), d in S.items():
        if d["arm"] != arm or d["seed"] != s:
            bad.append(f"{arm}_s{s}/summary.json says {d['arm']} seed {d['seed']}")
    if bad:
        raise SystemExit("REFUSED: " + "; ".join(bad))
    return S


def wm1(eval_dir, paired):
    Z = {(a, s): np.load(os.path.join(eval_dir, f"eval_{a}_s{s}.npz")) for a in ARMS for s in SEEDS}
    eid = Z[("M1", 0)]["eid"]
    for k, z in Z.items():
        if not np.array_equal(z["eid"], eid):
            raise SystemExit(f"REFUSED: eval_{k[0]}_s{k[1]} is not on the same windows")
    mean = {k: float(z["ade6"].mean()) for k, z in Z.items()}
    F = {a: abs(mean[(a, 0)] - mean[(a, 1)]) for a in ARMS}
    Fmax = max(F.values())
    ci = []
    for s in SEEDS:
        r = paired(Z[("M2b", s)]["ade6"], Z[("M1", s)]["ade6"], eid, n_boot=B, seed=0)
        ci.append({"seed": s, "mean": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"])})
    ok = all(c["separated"] and c["mean"] < 0 and abs(c["mean"]) > 2 * Fmax for c in ci)
    return {"definition": "M2b - M1, ADE 6 s (m), paired per seed; PASS iff separated at both seeds, mean < 0, |delta| > 2F, F = max(F_M1, F_M2b)",
            "per_seed": ci, "seed_floor_F_m": {**F, "max": Fmax}, "arm_mean_ade6_m": {f"{a}_s{s}": v for (a, s), v in mean.items()},
            "pass": bool(ok), "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid)))}


def families(eval_dir):
    out = {}
    for a in ARMS:
        for s in SEEDS:
            raw = json.load(open(os.path.join(eval_dir, f"eval_{a}_s{s}.json"), encoding="utf-8"))["raw"]
            out[f"{a}_s{s}"] = {k: raw.get(k) for k in ("traj", "families", "strategic", "tactical", "consistency_a",
                                                         "self_consistency_b", "acts") if k in raw}
    fr = os.path.join(eval_dir, "floors_report.json")
    if os.path.exists(fr):
        out["_floors"] = json.load(open(fr, encoding="utf-8")).get("floors")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        from taniteval.ci import paired_episode_cluster_bootstrap
    except ImportError:                                  # run from the repo checkout: <repo>/<pkg path>/code/this file
        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "taniteval"))
        from taniteval.ci import paired_episode_cluster_bootstrap
    S = load_runs(a.runs_dir)
    cmp_ = json.load(open(os.path.join(a.eval_dir, "compare.json"), encoding="utf-8"))
    k1 = {arm: {kk: cmp_["arms"][arm].get(kk) for kk in ("K1_ade6_minus_P10", "seed_floor_F_m", "K1_pass")} for arm in ARMS}
    k1_pass = any(bool(k1[arm]["K1_pass"]) for arm in ARMS)
    w = wm1(a.eval_dir, paired_episode_cluster_bootstrap)
    if k1_pass:
        nxt_k = "K-1 PASS => AR-1 (trajectory part) is met on this rig; the controls M2a / M1-navonly / M1-nocons are next (PI go)"
    else:
        nxt_k = "K-1 FAIL => plan 13.6 row 1: the same arms at --grid pool8x20 (cache_full holds it; queue stage train ARM SEED pool8x20, PI go)"
    nxt_w = ("WM-1 PASS => M2a x2 attributes the gain between the WM and its IDM term (PI go)" if w["pass"] else
             "WM-1 FAIL => the H-JEPA WM + IDM auxiliary is not a lever for v8F-mini at this scale; the WM's remaining route is rung beta "
             "(the WM as a PLANNER, not an auxiliary)")
    out = {"prereg": "PREREG_V8F_MINI Amendment 4", "tier": "T1-class open loop (eval-141, 2,399 windows)",
           "estimator": "paired episode-cluster bootstrap, B=2000, 141 episodes: answers ANOTHER DRAW OF EPISODES only; training-seed "
                        "variance enters only through F (2 seeds)",
           "provenance": {"train_sha256": S[("M1", 0)]["train_sha256"], "excluded_sha256": HOLDOUT_SHA, "n_clips": N_TRAIN,
                          "steps": S[("M1", 0)]["steps"], "S": S[("M1", 0)]["S"]},
           "K1": {"per_arm": k1, "pass": k1_pass, "multiplicity": "2 arms read, no selection on eval-141; PASS iff >= 1 arm passes"},
           "WM1": w, "P10_consistency_a_rate": cmp_.get("P10_consistency_a_rate"),
           "strategic_tactical_from_compare": {k: cmp_.get(k) for k in ("nav_used_M1navshuf_minus_M1_pos_err", "tactical_lat", "tactical_lon")},
           "families": families(a.eval_dir), "next": [nxt_k, nxt_w]}
    out["summary"] = (f"K-1 {'PASS' if k1_pass else 'FAIL'} (M1 {k1['M1']['K1_pass']}, M2b {k1['M2b']['K1_pass']}); "
                      f"WM-1 {'PASS' if w['pass'] else 'FAIL'} (M2b-M1 s0 {w['per_seed'][0]['mean']:+.4f}, s1 {w['per_seed'][1]['mean']:+.4f}, "
                      f"2F {2 * w['seed_floor_F_m']['max']:.4f})")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(out["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
