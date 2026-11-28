#!/usr/bin/env python3
"""CPU labeller of Step A: the true PDM sub-scores of every PERTURBED plan, on the instrument that labelled P2's training sets and the banked held-out table
(`refe/navsim_pdm_targets.py`: NAVSIM v1.1 @ 3e8291b's own MetricCacheProcessor + PDMSimulator + PDMScorer on a scenario rebuilt from the nuPlan DB).  Derived from
`raw/2026-10-18-p2-heldout-table/code/p2ho_score.py`; the difference is only the plan array ([Q,20,3] executed perturbations per token instead of the 64 banked proposals).

G1 label identity (prereg section 2), run FIRST and enforced: `--g1 N` labels the 64 banked in-fan proposals of N sampled tokens and requires <= 1e-6 against the banked table; its
control shifts every proposal 0.5 m forward and requires the label to CHANGE.  Exit 3 on either, and the main labelling is not started.

UNTESTED on data in this package (needs the navsim / nuPlan venv; CPU only).  The pure parts (`g1_check`, the control) are unit-tested in test_stepA.py.

    navsim-crun label_perturbations.py --bundle <bundle.npz> --perts <perts.npz> --plan <heldout_tokens.json> --table <table_p2_heldout.npz> --out <labels.npz> --g1 20 [--tree <PK>]
    env as p2ho_score.py: PYTHONPATH=<navsim v1.1>;<tree>/refe  NUPLAN_MAPS_ROOT=...  NUPLAN_MAP_VERSION=nuplan-maps-v1.0  REFE_SIM_HZ=10  OMP_NUM_THREADS=1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stepA as A  # noqa: E402

SUBS = A.SUBS


def label_plans(PT, parts, exec20):
    """[Q,20,3] executed perturbations -> (sub [Q,6] float64, pdms [Q]).  `executed_plans8` is the labeller's own conversion (A7 repair, then to_navsim); exec20 is already repaired, which is idempotent."""
    res = PT.score_plans(parts, PT.executed_plans8(np.asarray(exec20, np.float64), repair=True))
    return np.stack([np.asarray(res[k], np.float64) for k in SUBS], -1), np.asarray(res["pdms"], np.float64)


def g1(PT, NS, plan, table, tokens, n, seed=A.SEED):
    """G1 + control on n deterministic tokens: banked proposals [64,8,3] relabelled vs the banked sub-scores and pdms."""
    t = np.load(table, allow_pickle=True)
    ix = {str(k): i for i, k in enumerate(t["token"])}
    pick = sorted(np.random.default_rng(seed).choice(len(tokens), size=min(n, len(tokens)), replace=False).tolist())
    worst, ctrl_changed, checked = 0.0, 0, []
    by_log = {}
    for j in pick:
        by_log.setdefault(plan["token_log"][tokens[j]], []).append(tokens[j])
    for log, lt in sorted(by_log.items()):
        scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(plan["db"][log], sorted(lt), history_rows=40, future_rows=100)}
        for tk in sorted(lt):
            if tk not in scs:
                continue
            parts = PT.metric_cache_parts(scs[tk])
            P8 = np.asarray(t["proposals"][ix[tk]], np.float32)
            res = PT.score_plans(parts, P8)
            sub = np.stack([np.asarray(res[k], np.float64) for k in SUBS], -1)
            ok = np.asarray(t["valid"][ix[tk]], bool)
            worst = max(worst, A.g1_check(np.concatenate([sub[ok], np.asarray(res["pdms"], np.float64)[ok, None]], -1),
                                          np.concatenate([np.asarray(t["sub"][ix[tk]], np.float64)[ok], np.asarray(t["pdms"][ix[tk]], np.float64)[ok, None]], -1))["max_abs_diff"])
            sh = P8.copy()
            sh[..., 0] += 0.5                                            # control: every proposal 0.5 m further ahead must change SOME label
            ctrl = PT.score_plans(parts, sh)
            ctrl_changed += int(not np.allclose(np.asarray(ctrl["pdms"]), np.asarray(res["pdms"]), atol=1e-9))
            checked.append(tk)
    return {"n_checked": len(checked), "max_abs_diff": worst, "tol": 1e-6, "passed": bool(len(checked) > 0 and worst <= 1e-6), "control_tokens_whose_label_changed": ctrl_changed,
            "control_ok": bool(ctrl_changed > 0), "tokens": checked}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--perts", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g1", type=int, default=20)
    ap.add_argument("--tree", default="C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.tree, "refe"))
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    PT._setup()
    b, pe = A.load_npz(a.bundle), A.load_npz(a.perts)
    plan = json.load(open(a.plan, encoding="utf-8"))
    toks = [str(t) for t in b["token"]]
    g = g1(PT, NS, plan, a.table, toks, a.g1)
    json.dump(g, open(a.out + ".g1.json", "w"), indent=1)
    print("ZZG1", json.dumps({k: g[k] for k in g if k != "tokens"}), flush=True)
    if not (g["passed"] and g["control_ok"]):
        print("ZZLABEL_FAIL G1 (identity or control)", flush=True)
        return 3
    N, Q = pe["anchor"].shape
    sub = np.full((N, Q, 6), np.nan)
    pdms = np.full((N, Q), np.nan)
    by_log = {}
    for i, t in enumerate(toks):
        by_log.setdefault(plan["token_log"][t], []).append(i)
    t0 = time.time()
    for log, rows in sorted(by_log.items()):
        want = sorted(toks[i] for i in rows)
        scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(plan["db"][log], want, history_rows=40, future_rows=100)}
        for i in rows:
            sc = scs.get(toks[i])
            if sc is None:
                print(f"  {log}: scenario builder skipped {toks[i]}", flush=True)
                continue
            sub[i], pdms[i] = label_plans(PT, PT.metric_cache_parts(sc), pe["exec"][i])
        print(f"  {log}: {len(rows)} tokens, {time.time() - t0:.0f} s", flush=True)
    tmp = a.out + f".tmp{os.getpid()}.npz"
    np.savez(tmp, sub=sub, pdms=pdms, sub_names=np.array(SUBS), perts_sha256=np.array(A.sha256_file(a.perts)))
    os.replace(tmp, a.out)
    ok = bool(np.isfinite(sub).all() and np.isfinite(pdms).all())
    print(f"ZZLABEL_{'OK' if ok else 'PARTIAL'} {int(np.isfinite(pdms).all(1).sum())}/{N} {time.time() - t0:.0f} s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
