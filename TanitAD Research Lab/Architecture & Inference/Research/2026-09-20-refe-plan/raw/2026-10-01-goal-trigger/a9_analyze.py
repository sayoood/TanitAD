#!/usr/bin/env python3
"""SPEC_NAVTEST Amendment 9 -- the registered analysis (PREREG_AMENDMENT_9.md). Validity gates (a)-(e) first; then per
arm the statistic (per-token PDMS(arm) - PDMS(OFF), paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed
20260927) read PRIMARY (every confirmation token) and FRESH-LOG (tokens in logs holding no selection token), the
full-navtest effect (sum of deltas / 12,146), the four families, and the registered decision. Every number is read from
an artifact. Writes a9/result_a9.json and RESULT_A9_GOAL_FIX.md."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
DRV = os.environ.get("REFE_DRIVE", "D:")
sys.path.insert(0, str(PKG / "eval"))
import proxy_eval as PE  # noqa: E402

A9 = HERE / "a9"
DATA = Path(f"{DRV}/Projects/TanitAD/data/refe_navtest")
OFF_SEAM = DATA / "seams" / "refe_navtest_final.npz"
OFF_CSV = DATA / "score" / "refe_navtest_final" / "refe_navtest_final.csv"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SEL = HERE.parent / "2026-09-28-m6b-tangent" / "tokens_1123.json"
TANITAD_PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EV6 = ("C:/Users/Admin/ev6/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code")
GATING = ("pdm_route", "navgoal_straight")
ARMS = ("pdm_route", "navgoal_straight", "navgoal_arc", "a8_lane")
N_NAVTEST = 12146


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def families(seam_npz: Path, toks: list, label: str) -> dict | None:
    """families6 on a token subset of a seam (a subset npz is written so OFF and arm read the SAME tokens)."""
    z = np.load(seam_npz)
    idx = [i for i, t in enumerate(z["token"]) if str(t) in set(toks)]
    sub = A9 / "families" / f"{label}_seam.npz"
    sub.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sub, token=z["token"][idx], fingerprint=z["fingerprint"][idx], poses=z["poses"][idx],
             sampling=z["sampling"], arm=z["arm"])
    out = A9 / "families" / f"{label}.json"
    subprocess.call([TANITAD_PY, "families6.py", "--seam", str(sub), "--inputs", EXPORT, "--stage", "1",
                     "--label", label, "--out", str(out), "--n-boot", "2000"], cwd=EV6,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    stdout=open(A9 / "families" / f"{label}.log", "w"), stderr=subprocess.STDOUT)
    return jl(out)


def main() -> int:
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: E[t]["log_name"] for t in E}
    sel_logs = {tl[t] for t in json.load(open(SEL, encoding="utf-8"))["tokens"]}
    G = jl(HERE / "goal_fix_census.json")
    gsum = G["summary"]
    off_seam = np.load(OFF_SEAM)
    off_pose = {str(t): off_seam["poses"][i] for i, t in enumerate(off_seam["token"])}
    off_sc = {t: 100.0 * float(r["score"]) for t, r in PE.read_csv(OFF_CSV).items()}
    res = {"amendment": "SPEC_NAVTEST Amendment 9 (PREREG_AMENDMENT_9.md)", "off_seam": str(OFF_SEAM),
           "off_csv": str(OFF_CSV), "off_csv_rows": len(off_sc),
           "estimator": "paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927",
           "tier": "NAVSIM v1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents", "arms": {}}
    for arm in ARMS:
        ts = jl(A9 / f"tokens_{arm}.json")
        seam_p = A9 / f"seam_{arm}.npz"
        csv_p = DATA / "score" / f"refe_a9_{arm}" / f"refe_a9_{arm}.csv"
        if not (ts and seam_p.exists() and csv_p.exists()):
            res["arms"][arm] = {"missing": [str(p) for p in (seam_p, csv_p) if not p.exists()] or ["tokens"]}
            continue
        conf, ctrl = ts["confirm"], ts["controls"]
        z = np.load(seam_p)
        pose = {str(t): z["poses"][i] for i, t in enumerate(z["token"])}
        rec = jl(A9 / f"inputs_{arm}.json") or {}
        sc = {t: 100.0 * float(r["score"]) for t, r in PE.read_csv(csv_p).items()}
        rep = jl(seam_p.with_suffix(".report.json")) or {}
        st = jl(DATA / "score" / f"refe_a9_{arm}" / f"refe_a9_{arm}.status.json")
        gates = {}
        # (a) the goal changes on every confirmation token, in the run's own recorded inputs
        census_none = {t: G["rows"][t]["None"]["goal"] for t in conf if t in G["rows"] and "None" in G["rows"][t]}
        ch = [t for t in conf if t in rec and np.abs(np.asarray(rec[t]["goal"]) - np.asarray(census_none.get(t, rec[t]["goal"]))).max() > 0]
        gates["a"] = {"ok": len(ch) == len(conf), "goal_changed": f"{len(ch)}/{len(conf)}", "recorded": len(rec)}
        # (b) the regression control: goal_fix=None reproduces the pre-patch goals
        rg = gsum["regression_none_vs_prepatch"]
        gates["b"] = {"ok": rg["n"] == N_NAVTEST and rg["max_abs"] == 0.0, **rg}
        # (c) the 24 controls reproduce the OFF seam bit for bit
        eq = [bool(np.array_equal(pose[t], off_pose[t])) for t in ctrl if t in pose]
        gates["c"] = {"ok": len(eq) == len(ctrl) and all(eq), "controls_identical": f"{sum(eq)}/{len(ctrl)}"}
        # (d) the harness PASSes with every token valid; the seam has every row
        gates["d"] = {"ok": len(sc) == len(ts["tokens"]) == rep.get("rows") and rep.get("misses") == 0,
                      "csv_rows": len(sc), "seam_rows": rep.get("rows"), "seam_misses": rep.get("misses"),
                      "frame_control_max_m": (rep.get("frame_control") or {}).get("max_m"), "status_file": st}
        # (e) cross-check (pdm_route only; reported)
        if arm == "pdm_route":
            gates["e_crosscheck_reported"] = gsum.get("crosscheck_pdm_route_vs_metric_cache_centerline")
        ok = all(v["ok"] for k, v in gates.items() if isinstance(v, dict) and "ok" in v)
        reads = {}
        seen = set(ts["seen_logs"])                    # logs where the arm's defect was seen in exploration (as A8)
        for name, sel in (("PRIMARY", conf), ("FRESH-LOG", [t for t in conf if tl[t] not in seen])):
            sel = [t for t in sel if t in sc and t in off_sc]
            D = {t: sc[t] - off_sc[t] for t in sel}
            mu, lo, hi = PE.boot(D, tl) if D else (float("nan"),) * 3
            reads[name] = {"n_tokens": len(sel), "n_logs": len({tl[t] for t in sel}), "D_mean": mu, "ci95": [lo, hi],
                           "pdms_off": float(np.mean([off_sc[t] for t in sel])) if sel else None,
                           "pdms_arm": float(np.mean([sc[t] for t in sel])) if sel else None,
                           "n_off_zero": int(sum(off_sc[t] == 0 for t in sel)), "n_arm_zero": int(sum(sc[t] == 0 for t in sel))}
        D_all = sum(sc[t] - off_sc[t] for t in conf if t in sc and t in off_sc)
        full = {"delta_full_navtest_pdms": D_all / N_NAVTEST, "pdms_off_full": float(np.mean(list(off_sc.values()))),
                "note": "only confirmation tokens change; selection-token and frame-fail tokens are NOT in this sum"}
        arm_res = {"gates": gates, "gates_ok": ok, "reads": reads, "full_navtest": full,
                   "n_confirm": len(conf), "n_controls": len(ctrl)}
        if arm in GATING:
            fo = families(OFF_SEAM, conf, f"a9_off_for_{arm}")
            fa = families(seam_p, conf, f"a9_{arm}")
            adv = PE.fam_adverse(fo, fa) if fo and fa else {"missing": True}
            arm_res["families_adverse_arm_vs_off"] = adv
            P, F = reads["PRIMARY"], reads["FRESH-LOG"]
            no_adv = not adv.get("missing") and not adv.get("adverse")
            if not ok:
                v = "NOT PROVEN (gate failed: " + ", ".join(k for k, g in gates.items() if isinstance(g, dict) and not g.get("ok", True)) + ")"
            elif P["ci95"][0] > 0 and F["ci95"][0] > 0 and no_adv:
                v = "ADOPT"
            elif P["ci95"][1] < 0:
                v = "REFUTED"
            else:
                v = "NOT PROVEN"
            arm_res["verdict"] = v
        else:
            arm_res["verdict"] = "REPORTED (not gating)"
        res["arms"][arm] = arm_res
        print(arm, json.dumps({"gates_ok": ok, "verdict": arm_res["verdict"],
                               "PRIMARY": reads["PRIMARY"], "FRESH": reads["FRESH-LOG"], "full": full["delta_full_navtest_pdms"]}), flush=True)
    json.dump(res, open(A9 / "result_a9.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    print("ZZA9_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
