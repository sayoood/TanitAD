"""PREREG_RA1_FULL Amendment 1's readings (the K-PASS lever), committed before its data.  Pure JSON + perwindow.npz, no torch.

  python ra1_full_amend1.py --root <logs/ra1_full> --verdict <full4_verdict.json> --S 3000 --out amend1_final.json

  1. SENS   -- copied from the queue verdict's SENS block over R0/A0/A1/A1x/A2 at S (PREREG_HJ_STAGE0 Amendment 3's yaw-rate bar:
               A1 level-1 yaw_rate k=10 >= 0.10 AND A1 - A0 separated > 0, at BOTH seeds).  A seed the verdict could not evaluate is
               REFUSED, never read as a FAIL.
  2. REPLICATE -- the gate's training-seed replicate: A1x_s1 - A1_s1, eval-141 ADE@6s, paired over episodes on window-aligned
               arrays (ra1_full_report.paired_runs): > 0 AND separated => REPLICATES.  The holdout difference is reported, decides nothing.
  3. A2     -- the K arm re-picked on the HOLDOUT among R0, A1, A2 at S (lowest seed mean; within 0.01 m -> the simpler, R0 < A1 < A2).
               If A2 is picked, K is read for A2: A2 - P10-yaw-4512 < 0, separated at both seeds, |delta| > 2 x max(F_A1, F_A2) (the
               stage-0 bar names A1's floor; the larger of the two is never easier).  A2 - A1 per seed is descriptive.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (episodes); F = |s0 - s1| on eval-141 (training seed).
Exit code: 0 when the inputs parse (the readings are in the JSON); 2 on a missing / malformed / inconsistent input."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

TIE_M = 0.01
CELL = "b0.08g50"
RUNDIR = {"R0": "R0_s{s}", "A0": "A0_b0.08_s{s}", "A1": "A1_" + CELL + "_s{s}", "A1x": "A1x_" + CELL + "_s{s}", "A2": "A2_" + CELL + "_s{s}"}
ORDER = ("R0", "A1", "A2")                      # simplest first: a holdout tie goes to the earlier arm


def _report():
    p = Path(__file__).resolve().parent / "ra1_full_report.py"
    spec = importlib.util.spec_from_file_location("ra1_full_report", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read(a, pairfn) -> dict:
    rep = _report()
    base = os.path.join(a.root, "runs", f"st{a.S}")
    runs = {}
    for arm, tmpl in RUNDIR.items():
        for s in (0, 1):
            r = rep.load_run(os.path.join(base, tmpl.format(s=s)))
            if r is not None:
                runs[(arm, s)] = r
    need = [(arm, s) for arm in ("R0", "A0", "A1", "A2") for s in (0, 1)] + [("A1x", 1)]
    miss = [f"{arm}_s{s}" for arm, s in need if (arm, s) not in runs]
    if miss:
        raise ValueError(f"Amendment 1 needs every run at st{a.S}; missing {miss}")
    rep.check_full({a.S: {arm: {s: r for (ar, s), r in runs.items() if ar == arm} for arm in RUNDIR}})
    with open(a.verdict, encoding="utf-8") as f:
        v = json.load(f)
    if v.get("kin_mode") != "yaw" or not v.get("deciding") or not v.get("instrument_ok_all"):
        raise ValueError("the full4 verdict is not a deciding, instrument-OK yaw-rate verdict")
    out = dict(schema="ra1-full-amend1-v1", prereg="PREREG_RA1_FULL.md Amendment 1", S=a.S, cell=CELL,
               tier="T1-class open-loop trajectory (SENS is T0)", estimator="paired episode-cluster bootstrap, B=2000 (episodes); F = |s0 - s1|")
    # ---- 1. SENS
    sens = v.get("SENS") or {}
    per = sens.get("per_seed") or {}
    for s in ("seed0", "seed1"):
        if not isinstance(per.get(s), dict):
            raise ValueError(f"SENS {s} is not evaluable in the verdict: {per.get(s)!r} (a missing run is never a FAIL)")
    out["SENS"] = dict(bar=sens.get("bar"), PASS=bool(sens.get("PASS")), per_seed=per)
    # ---- 2. the gate's training-seed replicate
    d141 = rep.paired_runs(runs[("A1x", 1)], runs[("A1", 1)], "eval141", "ade6", "ade6", pairfn)
    dh = rep.paired_runs(runs[("A1x", 1)], runs[("A1", 1)], "holdout", "hold_ade6", "hold_ade6", pairfn)
    rep_ok = bool(d141["delta"] > 0 and d141["separated"])
    out["REPLICATE"] = dict(quantity="A1x_s1 - A1_s1, eval-141 ADE@6s", eval141=d141, holdout_NON_DECIDING=dh,
                            verdict="REPLICATES" if rep_ok else "DOES NOT REPLICATE")
    # ---- 3. A2: holdout re-pick, K for A2 if picked, A2 - A1 descriptive
    hold = {arm: sum(runs[(arm, s)]["ev"]["holdout"]["ade6_arm"]["mean"] for s in (0, 1)) / 2 for arm in ORDER}
    m = min(hold.values())
    k_arm = next(arm for arm in ORDER if hold[arm] <= m + TIE_M)
    e = {arm: [runs[(arm, s)]["ev"]["K"]["ade6_arm"]["mean"] for s in (0, 1)] for arm in ("A1", "A2")}
    F = {arm: abs(e[arm][0] - e[arm][1]) for arm in e}
    k2 = []
    for s in (0, 1):
        dk = runs[("A2", s)]["ev"]["K"]["ade6_arm_minus_p10"]
        k2.append(dict(seed=s, arm_minus_p10=dk, below=dk["delta"] < 0, separated=bool(dk["separated"]),
                       beats_2x_floor=abs(dk["delta"]) > 2 * max(F.values())))
    a2_pass = all(r["below"] and r["separated"] and r["beats_2x_floor"] for r in k2)
    out["A2"] = dict(holdout_seed_mean_ade6=hold, K_arm=k_arm, floors_eval141=F, two_F=2 * max(F.values()),
                     K_for_A2=dict(read=k_arm == "A2", PASS=a2_pass if k_arm == "A2" else None, rows=k2,
                                   note=None if k_arm == "A2" else "A2 not the holdout's pick: its K rows are DESCRIPTIVE"),
                     A2_minus_A1=[dict(seed=s, eval141=rep.paired_runs(runs[("A2", s)], runs[("A1", s)], "eval141", "ade6", "ade6", pairfn),
                                       holdout=rep.paired_runs(runs[("A2", s)], runs[("A1", s)], "holdout", "hold_ade6", "hold_ade6", pairfn))
                                  for s in (0, 1)])
    # ---- the branch
    if out["SENS"]["PASS"]:
        nxt = ("SENS PASS + K PASS => PREREG_HJ_STAGE0: frozen + adapter suffices; build refav2 (RA-2) -- its design is a Master Mind / "
               "PI step, not pre-registered here.")
    else:
        nxt = ("SENS FAIL => PREREG_HJ_STAGE0's ladder, run in order, each A1 x2 at S under the same SENS bar: gamma 200, then IDM at "
               "level 2 only, then a time-repulsion floor (|lambda_sim| <= 0.06).")
    if not rep_ok:
        nxt += (" The gate does NOT replicate at seed 1: every claim that TRUE time order adds skill is one-seed only; next is Addendum D's "
                "A1a at full scale.")
    out["next"] = nxt
    out["reading"] = (f"SENS {'PASS' if out['SENS']['PASS'] else 'FAIL'}; gate replicate {out['REPLICATE']['verdict']} "
                      f"(A1x_s1 - A1_s1 {d141['delta']:+.4f} [{d141['lo']:+.4f}, {d141['hi']:+.4f}]); holdout K arm {k_arm}"
                      + (f", K for A2 {'PASS' if a2_pass else 'FAIL'}" if k_arm == "A2" else "") + ".")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        from taniteval.ci import paired_episode_cluster_bootstrap
    except ImportError:                                  # run from the repo checkout: <repo>/<pkg path>/code/this file
        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "taniteval"))
        from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        res = read(a, paired_episode_cluster_bootstrap)
    except (OSError, KeyError, ValueError) as e:
        print(f"AMEND1 INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
