#!/usr/bin/env python3
"""r0c_ba_read.py -- B minus A on RL-val at ONE check, with the interval r0c_read.py needs.  (GAP 0: the chain does not write this.)

refe/rls1_pod.sh 9d reads  A0=base CTRL=arm A R=arm B --base A0  once, at u1000.  That file carries (A - A0) and (B - A0), each with an interval, but NOT B - A: the interval of a difference of two
paired differences needs the per-set deltas, and the JSON keeps only counts.  This writes the missing file, with the chain's own functions:

    python r0c_ba_read.py --a <arm>/scores/scores_update500_A.npz --b <arm>/scores/scores_update500_B.npz --pdm <pdm_val> --terms <terms_val> \
        [--rl-val-scenes <tsv>] --out <arm>/rlval_BvsA_update500.json            (run it for update250 / 500 / 750 / 1000; the egress glob rlval_*.json banks them)

Same estimator as every other RL-val read (rl_val_eval.run: paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927); the scorer dumps are read by rls1_checks.load_scores_light (one read per column:
rl_val_eval.load_scores holds ~30 GiB per file at N = 3,000).  The arms are NAMED A and B and the base is A: r0c_read.py refuses any file that is not that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pdm", required=True)
    ap.add_argument("--terms", required=True)
    ap.add_argument("--rl-val-scenes", default=None)
    ap.add_argument("--rule", default="navsim_v1_lane")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--pkg", default=os.path.normpath(os.path.join(HERE, "..", "..")), help="the refe-plan package root (eval/ and refe/ under it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for sub in ("eval", "refe"):
        sys.path.insert(0, os.path.join(a.pkg, sub))
    import rl_val_eval as RV
    import rl_reward
    import rls1_checks as RC
    arms = {"A": RC.load_scores_light(a.a), "B": RC.load_scores_light(a.b)}
    pdm, pst = RV.load_pdm(a.pdm)
    terms, tst = RV.load_terms(a.terms)
    cfg = rl_reward.RewardCfg(lam_d=0.15, w_prog=0.5, guard=True, use_k=False, use_l=False)
    scenes = RV.read_scenes(a.rl_val_scenes) if a.rl_val_scenes else None
    try:
        res = RV.run(arms, pdm, terms, None, a.rule, cfg, scenes, scenes is not None, "A", a.n_boot)
    except ValueError as e:
        print(f"ZZBA_REFUSED {e}", flush=True)
        return 3
    res["inputs"] = {"scores_sha256": {"A": RV.sha256_file(a.a), "B": RV.sha256_file(a.b)}, "pdm_dir": os.path.basename(os.path.normpath(a.pdm)), "pdm_stats": pst, "terms_stats": tst,
                     "rl_val_scenes_sha256": RV.sha256_file(a.rl_val_scenes) if a.rl_val_scenes else None, "tool": "r0c_ba_read/1"}
    res["rl_val_scenes_checked"] = scenes is not None
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    sc = res["selfcheck"]
    ok = sc["selfpair"]["ok"] and sc.get("swap", {"ok": True})["ok"]
    p = res["paired_vs_base"]["B"]["primary"]
    print(f"ZZBA_{'OK' if ok else 'SELFCHECK_FAILED'} sets {res['eligibility']['sets_used']} PDMS B-A {p['mean']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]", flush=True)
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
